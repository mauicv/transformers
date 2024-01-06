import torch
import time
import string
import random

alphabet = string.ascii_lowercase + string.digits
def uuid(length=4):
    return ''.join(random.choices(alphabet, k=length))

class Profiling(object):
    def __init__(self, model):
        if isinstance(model, torch.nn.Module) is False:
            print("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self.record = {
            'forward':[],
        }
        self.profiling_on = True
        self.origin_call = {}
        self.hook_done = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        ret = ""
        ret += "\n================================= Profile =================================\n"
        ret += "\nFORWARD TIME:\n"

        ts = self.record['forward'][0][1]
        te = self.record['forward'][-1][1]
        ret += f"\nTotal time:\t{1000*(te - ts):.6f} ms\n\n"

        ret += ('-------------------\n')
        for i, ((name1, ts1, event1), (name2, ts2, event2)) in enumerate(zip(
                self.record['forward'],
                self.record['forward'][1:]
            )):
            ret += (
                f"event{i+1:3d}:\t{1000*(ts2 - ts1):10.6f} ms"
                f"\t({event1}:{name1} -> {event2}:{name2})\n"
            )

        ret += ('-------------------\n')
        component_time = 0
        for name, ts1, ts2 in self.component_events:
            diff = ts2 - ts1
            ret += (f"{1000*(diff):0.6f} ms \t ({name}) \n")
            component_time += diff

        ret += ('-------------------\n')
        ret += (f"{1000*(component_time):0.6f} ms \t (total-component-time) \n")    
        ret += (f"{1000*(te - ts - component_time):0.6f} ms \t (others) \n")    

        return ret

    def start(self):
        if self.hook_done is False:
            self.hook_done = True
            self.hook_modules(self.model, self.model.__class__.__name__)
        self.profiling_on = True
        return self

    @property
    def component_events(self):
        comp_data = {}
        component_names = []
        for component_name, ts, event in self.record['forward']:
            if component_name not in comp_data:
                comp_data[component_name] = {}
                component_names.append(component_name)
            comp_data[component_name][event] = ts

        for component_name in component_names:
            yield (
                component_name,
                comp_data[component_name]['start'],
                comp_data[component_name]['end'],
            )

    def stop(self):
        self.profiling_on = False
        return self

    def hook_modules(self, module, name):
        for name, layer in module.named_children():
            if isinstance(layer, torch.nn.ModuleList):
                for ind, sub_sub_module in enumerate(layer):
                    self._hook_module(f'{name}-{ind}', sub_sub_module)
            else:
                self._hook_module(name, layer)

    def _hook_module(self, name, layer):
        uid = uuid(length=4)
        name = name + '-' + uid
        def make_hook(event):
            def hook(layer, *args, **kwargs):
                t = time.time()
                if (self.profiling_on):
                    self.record['forward'].append(
                        (name, t, event)
                    )

            return hook
    
        layer.register_forward_hook(
            make_hook('end')
        )
        layer.register_forward_pre_hook(
            make_hook('start')
        )

