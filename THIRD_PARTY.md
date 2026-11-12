# Third-party code

`src/supermimic/env/mdp/geometry/{math,state,representation}.py` adapts MaxCoord
helpers and integration from [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic),
commit `5efbf114e11f725c31d83e217be987c13c0e4ce2`, under Apache-2.0.
The license is retained in `licenses/HybridRobotics-supermimic-Apache-2.0.txt`.
Changes replace project-specific imports and remove environment binding, actor,
reward, and IK responsibilities from the representation. Integration math and
loss channel definitions retain the upstream convention.
