"""Release automatically named FX sources after their compiled code dies.

Explicit filenames, including profiler content hashes, keep the original lazy
cache behavior because independently compiled code may share those names.
Forked children keep the inherited source snapshot unmanaged; new child sources
are owned normally. This avoids taking locks across independent at-fork handlers.
"""

import linecache
import os
import threading
import types
import weakref


_cache_lock = threading.RLock()
# These owners keep weakref callbacks alive until the last code reference dies.
_owned_sources = {}
_next_id = 0
_process_pid = os.getpid()


def _after_fork_child():
    global _cache_lock, _owned_sources, _process_pid
    pid = os.getpid()
    if pid != _process_pid:
        _cache_lock = threading.RLock()
        _owned_sources = {}
        _process_pid = pid


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


class _Lease:
    pass


class _Source:
    def __init__(self, key, source, entry):
        self.key = key
        self.source = source
        self.entry = entry
        self.references = {}


class OwnedEvalCacheLoader:
    def __init__(self):
        self.eval_cache = {}
        self.next_id = 0
        self._owners = {}

    @property
    def _lock(self):
        # Another library's earlier child handler can release code before ours.
        _after_fork_child()
        return _cache_lock

    def _get_key(self):
        global _next_id
        with self._lock:
            self.next_id = max(self.next_id, _next_id)
            key = f"<eval_with_key>.{self.next_id}"
            self.next_id += 1
            _next_id = self.next_id
            return key

    def cache(self, src, globals, co_fields=None):
        return self._cache(src, globals, co_fields, owned=False)[0]

    def _cache(self, src, globals, co_fields, *, owned):
        with self._lock:
            key = self._get_key()
            explicit_filename = False
            suffix = ""
            if co_fields and "co_filename" in co_fields:
                if "co_firstlineno" not in co_fields or "co_name" not in co_fields:
                    key = co_fields["co_filename"]
                    explicit_filename = True
                else:
                    suffix = (
                        f" from {co_fields['co_filename']}:"
                        f"{co_fields['co_firstlineno']} in {co_fields['co_name']}"
                    )
                    key += suffix
            if not explicit_filename:
                while (
                    key in self.eval_cache or key in linecache.cache
                    or key in _owned_sources
                ):
                    key = self._get_key() + suffix
            managed = (
                owned and not explicit_filename
                and key not in self.eval_cache and key not in linecache.cache
            )
            # Explicit aliases can refer to a previously generated key. Future
            # callbacks for that old generation must not reclaim the new entry.
            self._owners.pop(key, None)
            _owned_sources.pop(key, None)
            self.eval_cache[key] = src
            if not managed:
                globals_copy = globals.copy()
                globals_copy.update(__file__=key, __name__=key, __loader__=self)
                linecache.lazycache(key, globals_copy)
                return key, None, None

            # A unique owned entry has one stable identity, even after source
            # lookup. Shared/explicit filenames retain Torch's lazy semantics.
            entry = (len(src), None, [line + "\n" for line in src.splitlines()], key)
            linecache.cache[key] = entry
            owner = _Source(key, src, entry)
            self._owners[key] = owner
            _owned_sources[key] = owner
            lease = _Lease()
            self._track(owner, lease)
            return key, owner, lease

    def get_source(self, module_name):
        with self._lock:
            return self.eval_cache.get(module_name)

    def _track(self, owner, value):
        identity = id(value)
        previous = owner.references.get(identity)
        if previous is not None and previous() is value:
            return

        def released(reference):
            with self._lock:
                if owner.references.get(identity) is not reference:
                    return
                del owner.references[identity]
                if owner.references:
                    return
                if self._owners.get(owner.key) is owner:
                    del self._owners[owner.key]
                if _owned_sources.get(owner.key) is not owner:
                    return
                del _owned_sources[owner.key]
                if self.eval_cache.get(owner.key) is owner.source:
                    del self.eval_cache[owner.key]
                if linecache.cache.get(owner.key) is owner.entry:
                    del linecache.cache[owner.key]

        owner.references[identity] = weakref.ref(value, released)

    def _track_code_tree(self, owner, root):
        if owner is None:
            return
        with self._lock:
            pending = [root]
            visited = set()
            while pending:
                code = pending.pop()
                if id(code) in visited:
                    continue
                visited.add(id(code))
                self._track(owner, code)
                pending.extend(
                    value for value in code.co_consts if isinstance(value, types.CodeType)
                )

    def exec_with_source(self, src, globals, co_fields=None):
        key, owner, lease = self._cache(src, globals, co_fields, owned=True)
        # Keep the lease in this frame: a compile/exec traceback owns the source
        # until its frame dies, including SyntaxError where no code was created.
        code = compile(src, key, "exec")
        self._track_code_tree(owner, code)
        exec(code, globals)
