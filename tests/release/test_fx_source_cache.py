import concurrent.futures
import gc
import inspect
import linecache
import os
import subprocess
import sys
import traceback
import types
import unittest
import weakref

from scripts.environment.patches.torch_fx_source_cache import OwnedEvalCacheLoader


SOURCE = "def forward(value):\n    return value + 3\n"


class SourceOwnerTest(unittest.TestCase):
    def setUp(self):
        self.loader = OwnedEvalCacheLoader()
        # Distinct keys across test loaders preserve unrelated process entries.
        self.loader.next_id = id(self.loader)

    def assert_released(self, key):
        gc.collect()
        self.assertNotIn(key, self.loader.eval_cache)
        self.assertNotIn(key, linecache.cache)
        self.assertNotIn(key, self.loader._owners)

    def test_live_function_and_code_keep_source(self):
        namespace = {}
        self.loader.exec_with_source(SOURCE, namespace)
        function = namespace.pop("forward")
        key = function.__code__.co_filename
        self.assertEqual(inspect.getsource(function), SOURCE)
        code = function.__code__
        del function, namespace
        gc.collect()
        self.assertEqual(linecache.getlines(key), SOURCE.splitlines(keepends=True))
        del code
        self.assert_released(key)

    def test_equal_code_objects_have_independent_identity(self):
        left, right = {}, {}
        self.loader.exec_with_source(SOURCE, left)
        self.loader.exec_with_source(SOURCE, right)
        self.assertEqual(left["forward"].__code__, right["forward"].__code__)
        left_key = left["forward"].__code__.co_filename
        right_key = right["forward"].__code__.co_filename
        self.assertNotEqual(left_key, right_key)
        del left
        self.assert_released(left_key)
        self.assertEqual(inspect.getsource(right["forward"]), SOURCE)
        del right
        self.assert_released(right_key)

    def test_nested_function_outlives_factory(self):
        source = "def factory():\n    def inner(value):\n        return value + 9\n    return inner\n"
        namespace = {}
        self.loader.exec_with_source(source, namespace)
        inner = namespace["factory"]()
        key = inner.__code__.co_filename
        del namespace
        gc.collect()
        self.assertEqual(inner(1), 10)
        self.assertIn("def inner", inspect.getsource(inner))
        del inner
        self.assert_released(key)

    def test_generator_and_coroutine_frames_keep_source(self):
        for source, name in (
            ("def generate():\n    yield 4\n", "generate"),
            ("async def run():\n    return 4\n", "run"),
        ):
            namespace = {}
            self.loader.exec_with_source(source, namespace)
            function = namespace[name]
            key = function.__code__.co_filename
            active = function()
            del namespace, function
            gc.collect()
            self.assertIn(key, self.loader.eval_cache)
            active.close()
            del active
            self.assert_released(key)

    def test_retained_traceback_keeps_generated_source(self):
        source = "def fail():\n    raise ValueError('generated failure')\n"
        namespace = {}
        self.loader.exec_with_source(source, namespace)
        key = namespace["fail"].__code__.co_filename
        try:
            namespace["fail"]()
        except ValueError as error:
            retained = error.__traceback__
        del namespace
        gc.collect()
        self.assertIn("raise ValueError", "".join(traceback.format_tb(retained)))
        self.assertIn(key, self.loader.eval_cache)
        del retained
        self.assert_released(key)

    def test_compile_and_exec_failures_release_after_traceback(self):
        for source, error_type in (
            ("def bad(:\n", SyntaxError),
            ("raise RuntimeError('exec failure')\n", RuntimeError),
        ):
            namespace = {}
            before = set(self.loader.eval_cache)
            try:
                self.loader.exec_with_source(source, namespace)
            except error_type as error:
                retained = error.__traceback__
                rendered = "".join(traceback.format_exception(type(error), error, retained))
            keys = set(self.loader.eval_cache) - before
            self.assertEqual(len(keys), 1)
            key = keys.pop()
            self.assertIn(source.splitlines()[0], rendered)
            self.assertEqual(linecache.getlines(key), [line + "\n" for line in source.splitlines()])
            del retained, namespace
            self.assert_released(key)

    def test_origin_fields_retain_unique_ownership(self):
        namespace = {}
        self.loader.exec_with_source(SOURCE, namespace, {
            "co_filename": "original_model.py", "co_firstlineno": 17, "co_name": "forward",
        })
        key = namespace["forward"].__code__.co_filename
        self.assertIn(" from original_model.py:17 in forward", key)
        del namespace
        self.assert_released(key)

    def test_explicit_and_shared_keys_keep_existing_lazy_behavior(self):
        key = f"fx-owner-profiler-{id(self)}.py"
        fields = {"co_filename": key}
        first, second = {}, {}
        self.loader.exec_with_source(SOURCE, first, fields)
        self.assertEqual(len(linecache.cache[key]), 1)
        self.loader.exec_with_source(SOURCE, second, fields)
        del second
        gc.collect()
        self.assertEqual(inspect.getsource(first["forward"]), SOURCE)
        del first
        gc.collect()
        self.assertIn(key, self.loader.eval_cache)
        self.assertNotIn(key, self.loader._owners)
        # Distinct source with an unread shared key still resolves the latest.
        key2 = key + ".unread"
        self.loader.cache(SOURCE, {}, {"co_filename": key2})
        changed = SOURCE.replace("+ 3", "+ 7")
        self.loader.cache(changed, {}, {"co_filename": key2})
        self.assertEqual(linecache.getlines(key2), changed.splitlines(keepends=True))
        self.loader.cache(SOURCE, {}, fields)
        self.loader.cache(changed, {}, fields)
        self.assertEqual(linecache.getlines(key), SOURCE.splitlines(keepends=True))
        for filename in (key, key2):
            linecache.cache.pop(filename)

    def test_explicit_alias_invalidates_old_generation(self):
        namespace = {}
        self.loader.exec_with_source(SOURCE, namespace)
        key = namespace["forward"].__code__.co_filename
        changed = SOURCE.replace("+ 3", "+ 7")
        self.loader.cache(changed, {}, {"co_filename": key})
        del namespace
        gc.collect()
        self.assertEqual(self.loader.eval_cache[key], changed)
        self.assertIn(key, linecache.cache)
        self.assertNotIn(key, self.loader._owners)
        linecache.cache.pop(key)

    def test_foreign_replacements_are_not_deleted(self):
        namespace = {}
        self.loader.exec_with_source(SOURCE, namespace)
        key = namespace["forward"].__code__.co_filename
        foreign = (7, None, ["foreign\n"], key)
        linecache.cache[key] = foreign
        foreign_source = (SOURCE + "foreign")[:-7]
        self.assertEqual(foreign_source, SOURCE)
        self.assertIsNot(foreign_source, SOURCE)
        self.loader.eval_cache[key] = foreign_source
        del namespace
        gc.collect()
        self.assertIs(linecache.cache[key], foreign)
        self.assertIs(self.loader.eval_cache[key], foreign_source)
        linecache.cache.pop(key)

    def test_concurrent_compilations_and_lookups(self):
        def work(index):
            namespace = {}
            self.loader.exec_with_source(SOURCE, namespace)
            function = namespace["forward"]
            key = function.__code__.co_filename
            self.assertEqual(function(index), index + 3)
            self.assertEqual(inspect.getsource(function), SOURCE)
            return key

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            keys = list(executor.map(work, range(100)))
        gc.collect()
        self.assertEqual(len(set(keys)), 100)
        for key in keys:
            self.assert_released(key)

    def test_independent_loaders_allocate_distinct_live_keys(self):
        first_loader, second_loader = OwnedEvalCacheLoader(), OwnedEvalCacheLoader()
        first, second = {}, {}
        first_loader.exec_with_source(SOURCE, first)
        second_loader.exec_with_source(SOURCE, second)
        first_key = first["forward"].__code__.co_filename
        second_key = second["forward"].__code__.co_filename
        self.assertNotEqual(first_key, second_key)
        del first
        gc.collect()
        self.assertNotIn(first_key, linecache.cache)
        self.assertEqual(inspect.getsource(second["forward"]), SOURCE)
        del second
        gc.collect()
        self.assertNotIn(second_key, linecache.cache)

    def test_foreign_loader_explicit_alias_invalidates_owner(self):
        other = OwnedEvalCacheLoader()
        namespace = {}
        self.loader.exec_with_source(SOURCE, namespace)
        key = namespace["forward"].__code__.co_filename
        other.cache(SOURCE, {}, {"co_filename": key})
        del namespace
        gc.collect()
        self.assertEqual(linecache.getlines(key), SOURCE.splitlines(keepends=True))
        self.assertIn(key, other.eval_cache)
        linecache.cache.pop(key)

    def test_fork_registration_does_not_retain_loaders(self):
        temporary = OwnedEvalCacheLoader()
        reference = weakref.ref(temporary)
        del temporary
        gc.collect()
        self.assertIsNone(reference())

    def test_live_code_owns_temporary_loader_until_release(self):
        temporary = OwnedEvalCacheLoader()
        reference = weakref.ref(temporary)
        namespace = {}
        temporary.exec_with_source(SOURCE, namespace)
        key = namespace["forward"].__code__.co_filename
        del temporary
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertEqual(inspect.getsource(namespace["forward"]), SOURCE)
        del namespace
        gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn(key, linecache.cache)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_fork_recovers_lock_held_by_another_thread(self):
        script = '''
import gc, inspect, os, signal, threading, time
from scripts.environment.patches.torch_fx_source_cache import OwnedEvalCacheLoader
loader = OwnedEvalCacheLoader()
held = threading.Event()
def holder():
    with loader._lock:
        held.set()
        time.sleep(0.1)
worker = threading.Thread(target=holder)
worker.start()
assert held.wait(2)
child = os.fork()
if child == 0:
    signal.alarm(3)
    namespace = {}
    loader.exec_with_source("def forward(value):\\n    return value + 3\\n", namespace)
    assert "return value + 3" in inspect.getsource(namespace["forward"])
    del namespace
    gc.collect()
    assert not loader.eval_cache
    os._exit(0)
worker.join()
_, status = os.waitpid(child, 0)
assert os.waitstatus_to_exitcode(status) == 0, status
'''
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            capture_output=True, text=True, timeout=8,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_fork_preserves_snapshot_and_reclaims_new_sources(self):
        script = '''
import gc, inspect, linecache, os, signal, sys, threading
signal.alarm(8)
namespace = {}
def earlier_child_handler():
    namespace.clear()
    gc.collect()
if sys.argv[1] == "early":
    os.register_at_fork(after_in_child=earlier_child_handler)
from scripts.environment.patches.torch_fx_source_cache import OwnedEvalCacheLoader
if sys.argv[1] == "late":
    os.register_at_fork(after_in_child=earlier_child_handler)
loader = OwnedEvalCacheLoader()
source = "def forward(value):\\n    return value + 3\\n"
loader.exec_with_source(source, namespace)
live_namespace = {}
loader.exec_with_source(source, live_namespace)
live = live_namespace.pop("forward")
del live_namespace
held, release = threading.Event(), threading.Event()
def holder():
    with loader._lock:
        held.set()
        release.wait(4)
worker = threading.Thread(target=holder)
worker.start()
assert held.wait(2)
child = os.fork()
if child == 0:
    signal.alarm(3)
    assert inspect.getsource(live) == source
    del live
    gc.collect()
    baseline = set(loader.eval_cache)
    assert len(baseline) == 2
    for _ in range(20):
        generated = {}
        loader.exec_with_source(source, generated)
        assert inspect.getsource(generated["forward"]) == source
        del generated
        gc.collect()
        assert set(loader.eval_cache) == baseline
        assert baseline.issubset(linecache.cache)
    os._exit(0)
release.set()
worker.join()
_, status = os.waitpid(child, 0)
assert os.waitstatus_to_exitcode(status) == 0, status
del live
namespace.clear()
gc.collect()
assert not loader.eval_cache
'''
        for order in ("early", "late"):
            result = subprocess.run(
                [sys.executable, "-B", "-c", script, order],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, order + result.stdout + result.stderr)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_fork_does_not_reverse_another_library_lock_order(self):
        script = '''
import gc, os, signal, threading
from scripts.environment.patches.torch_fx_source_cache import OwnedEvalCacheLoader
signal.alarm(8)
loader = OwnedEvalCacheLoader()
runtime_lock = threading.RLock()
fx_held, runtime_held, finalizer_entered = (threading.Event() for _ in range(3))
class Finalizer:
    def __del__(self):
        finalizer_entered.set()
        with runtime_lock:
            pass
def holder():
    with loader._lock:
        value = Finalizer()
        value.cycle = value
        fx_held.set()
        assert runtime_held.wait(3)
        del value
        gc.collect()
worker = threading.Thread(target=holder)
worker.start()
assert fx_held.wait(2)
def before_runtime():
    runtime_lock.acquire()
    runtime_held.set()
    assert finalizer_entered.wait(3)
os.register_at_fork(before=before_runtime, after_in_parent=runtime_lock.release,
                    after_in_child=runtime_lock.release)
child = os.fork()
if child == 0:
    signal.alarm(3)
    generated = {}
    loader.exec_with_source("def forward(value):\\n    return value + 3\\n", generated)
    generated.clear()
    del generated
    gc.collect()
    assert not loader.eval_cache
    os._exit(0)
worker.join(3)
assert not worker.is_alive()
_, status = os.waitpid(child, 0)
assert os.waitstatus_to_exitcode(status) == 0, status
'''
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
