Module src.frx.helpers
======================

Functions
---------

`to_aim_value(value: object)`
:   Recursively converts objects into [Aim](https://github.com/aimhubio/aim)-compatible values.
    
    As a fallback, tries to call `to_aim_value()` on an object.

Classes
-------

`DummyAimRun(*args, **kwargs)`
:   

    ### Methods

    `track(self, metrics: dict[str, object], step: int)`
    :