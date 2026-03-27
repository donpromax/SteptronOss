import importlib
import inspect
import sys
import types


def get_func_name(func):
    if isinstance(func, str):
        return func
    return ".".join((func.__module__, func.__qualname__))


def dummy_function_wrapper(func_name):
    def dummy_function(*args, **kwargs):
        raise RuntimeError(f"function {func_name} no exist")

    return dummy_function


class Patch:
    def __init__(self, orig_func_name, new_func, create_dummy):
        split_name = orig_func_name.rsplit(".", 1)
        if len(split_name) == 1:
            self.orig_module_name, self.orig_func_name = orig_func_name, None
        else:
            self.orig_module_name, self.orig_func_name = split_name
        self.orig_module = None
        self.orig_func = None
        self.patch_func = None
        self.final_patch_func = None
        self.wrappers = []
        if new_func is None:
            new_func = dummy_function_wrapper(orig_func_name)
        self.set_patch_func(new_func)
        self.is_applied = False
        self.create_dummy = create_dummy

    def set_patch_func(self, new_func, force_patch=False):
        if hasattr(new_func, "__name__") and new_func.__name__.endswith(("wrapper", "decorator")):
            if new_func not in self.wrappers:
                self.wrappers.append(new_func)
        else:
            if self.patch_func and not force_patch:
                raise RuntimeError(f"the patch of {self.orig_func_name} exist !")
            self.patch_func = new_func
        self.is_applied = False

    def apply_patch(self):
        if self.is_applied:
            return

        current_module, current_func = Patch.parse_path(self.orig_module_name, self.orig_func_name, self.create_dummy)
        if self.orig_module is None:
            self.orig_module, self.orig_func = current_module, current_func

        final_patch_func = self.orig_func
        if self.patch_func is not None:
            final_patch_func = self.patch_func

        for wrapper in self.wrappers:
            final_patch_func = wrapper(final_patch_func)

        if self.orig_func_name is not None:
            setattr(self.orig_module, self.orig_func_name, final_patch_func)
        for _, value in sys.modules.copy().items():
            if (
                self.orig_func_name is not None
                and hasattr(value, self.orig_func_name)
                and getattr(value, self.orig_func_name) is current_func
            ):
                setattr(value, self.orig_func_name, final_patch_func)
        self.is_applied = True
        self.final_patch_func = final_patch_func

    @staticmethod
    def parse_path(module_path, function_name, create_dummy):
        from importlib.machinery import ModuleSpec

        modules = module_path.split(".")
        for i in range(1, len(modules) + 1):
            parent = ".".join(modules[: i - 1])
            path = ".".join(modules[:i])
            try:
                importlib.import_module(path)
            except ModuleNotFoundError as e:
                if not parent or not hasattr(importlib.import_module(parent), modules[i - 1]):
                    if not create_dummy:
                        raise ModuleNotFoundError(e) from e
                    sys.modules[path] = types.ModuleType(path)
                    sys.modules[path].__file__ = "steptronoss.dummy_module.py"
                    sys.modules[path].__spec__ = ModuleSpec(path, None)
                    if parent:
                        setattr(importlib.import_module(parent), modules[i - 1], sys.modules[path])
                else:
                    module = getattr(importlib.import_module(parent), modules[i - 1])
                    if hasattr(module, function_name):
                        return module, getattr(module, function_name)
                    if create_dummy:
                        return module, dummy_function_wrapper(function_name)
                    raise RuntimeError(f"no exist {function_name} of {module}")

        if function_name is not None and not hasattr(sys.modules[module_path], function_name):
            setattr(sys.modules[module_path], function_name, None)
        return sys.modules[module_path], getattr(
            sys.modules[module_path], function_name
        ) if function_name is not None else None


class StepTronPatchesManager:
    patches_info: dict[str, Patch] = {}

    @staticmethod
    def register_patch(orig_func_name, new_func=None, force_patch=False, create_dummy=False):
        if orig_func_name not in StepTronPatchesManager.patches_info:
            StepTronPatchesManager.patches_info[orig_func_name] = Patch(orig_func_name, new_func, create_dummy)
        else:
            StepTronPatchesManager.patches_info.get(orig_func_name).set_patch_func(new_func, force_patch)

    @staticmethod
    def apply_patches():
        for patch in StepTronPatchesManager.patches_info.values():
            patch.apply_patch()
