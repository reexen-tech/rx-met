"""Check FX source availability and reclamation in a separate bounded process."""


def main():
    import gc
    import inspect
    import linecache
    import traceback

    import torch

    import torch.fx.graph_module as graph_module

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    gc.collect()
    baseline = dict(graph_module._loader.eval_cache)

    def generated_function(index):
        namespace = {}
        graph_module._exec_with_source(
            f'def retained(value):\n    return value + {index}\n', namespace)
        return namespace.pop('retained')

    for index in range(20):
        function = generated_function(index)
        key = function.__code__.co_filename
        gc.collect()
        assert function(3) == index + 3
        assert f'return value + {index}' in inspect.getsource(function)
        assert key in graph_module._loader.eval_cache
        del function
        gc.collect()
        assert key not in graph_module._loader.eval_cache, 'dead generated function retained its source'
        assert key not in linecache.cache, 'dead generated function retained its linecache entry'

    def generated_traceback():
        namespace = {}
        graph_module._exec_with_source(
            "def fail():\n    raise ValueError('generated failure')\n", namespace)
        function = namespace.pop('fail')
        key = function.__code__.co_filename
        try:
            function()
        except ValueError as error:
            return error.__traceback__, key
        raise AssertionError('generated function did not raise')

    retained_traceback, key = generated_traceback()
    gc.collect()
    assert key in graph_module._loader.eval_cache
    assert "raise ValueError('generated failure')" in ''.join(traceback.format_tb(retained_traceback))
    del retained_traceback
    gc.collect()
    assert key not in graph_module._loader.eval_cache, 'dead traceback retained generated source'
    assert key not in linecache.cache, 'dead traceback retained its linecache entry'

    def failed_compile():
        try:
            graph_module._exec_with_source('def invalid(:\n', {})
        except SyntaxError:
            return
        raise AssertionError('invalid generated source did not raise')

    failed_compile()
    gc.collect()
    assert graph_module._loader.eval_cache == baseline, 'FX source cache grew after generated code was released'
    print(f'FX_SOURCE_CACHE_OK torch={torch.__version__}: live source, traceback and reclamation verified')


if __name__ == '__main__':
    main()
