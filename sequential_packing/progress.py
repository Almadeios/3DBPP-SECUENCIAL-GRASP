
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, *a, **k): 
        class Dummy:
            def __init__(self, it):
                self.it = it

            def __iter__(self):
                return iter(self.it)

            def update(self, *a, **k):
                pass

            def set_postfix(self, *a, **k):
                pass

            def close(self):
                pass

        return Dummy(x)


class SilentProgress:
    def update(self, *a, **k):
        pass

    def set_postfix(self, *a, **k):
        pass

    def close(self):
        pass
