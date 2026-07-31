from __future__ import annotations
from copy import deepcopy
from typing import Any,Dict,Optional
from curobo._src.util_file import load_yaml
def return_value_if_exists(input_dict:Dict,key:str,suffix:str="xrdf",raise_error:bool=True)->Any:
    if key not in input_dict:
        if raise_error:raise ValueError(f"{key} key not found in {suffix}")
        return None
    return input_dict[key]
def convert_xrdf_to_curobo(content_path=None,input_xrdf_dict:Optional[Dict]=None)->Dict:
    if input_xrdf_dict is None:
        path=getattr(content_path,"robot_xrdf_absolute_path",None)
        if path is None:raise ValueError("XRDF content or robot_xrdf_absolute_path is required")
        input_xrdf_dict=load_yaml(path)
    if input_xrdf_dict.get("format")!="xrdf":raise ValueError("format is not xrdf")
    cp=content_path; urdf=getattr(cp,"robot_urdf_absolute_path",None)
    x=input_xrdf_dict;c=x["cspace"];active=list(c["joint_names"]);defaults=x.get("default_joint_positions",{})
    kin={"urdf_path":urdf,"base_link":x.get("base_frame"),"tool_frames":deepcopy(x.get("tool_frames",{})),
         "collision_spheres":deepcopy(x.get("geometry",{}).get(x.get("collision",{}).get("geometry",""),{}).get("spheres",{})),
         "collision_sphere_buffer":x.get("collision",{}).get("buffer_distance",0.),"self_collision_ignore":deepcopy(x.get("self_collision",{}).get("ignore",{})),
         "self_collision_buffer":deepcopy(x.get("self_collision",{}).get("buffer_distance",{})),"lock_joints":{},
         "cspace":{"joint_names":active,"default_joint_position":[defaults.get(j,0.) for j in active],"null_space_weight":[1.]*len(active),"cspace_distance_weight":[1.]*len(active),"max_acceleration":list(c["acceleration_limits"]),"max_jerk":list(c["jerk_limits"])},
         "extra_links":{}}
    out={"robot_cfg":{"kinematics":kin}}
    if "dynamics" in x:out["robot_cfg"]["dynamics"]=deepcopy(x["dynamics"])
    return out
def convert_curobo_to_xrdf(input_curobo_dict:Dict,geometry_name:str="collision_model")->Dict:
    root=input_curobo_dict.get("robot_cfg",input_curobo_dict);k=root.get("kinematics",root)
    c=k.get("cspace",{});names=list(c.get("joint_names",[]));defaults=list(c.get("default_joint_position",[0.]*len(names)))
    return {"format":"xrdf","format_version":1.0,"geometry":{geometry_name:{"spheres":deepcopy(k.get("collision_spheres",{}))}},
      "collision":{"geometry":geometry_name,"buffer_distance":k.get("collision_sphere_buffer",0.)},
      "self_collision":{"geometry":geometry_name,"ignore":deepcopy(k.get("self_collision_ignore",{})),"buffer_distance":deepcopy(k.get("self_collision_buffer",{}))},
      "tool_frames":deepcopy(k.get("tool_frames",{})),"cspace":{"joint_names":names,"acceleration_limits":list(c.get("max_acceleration",[])),"jerk_limits":list(c.get("max_jerk",[]))},
      "default_joint_positions":dict(zip(names,defaults))}
__all__=["return_value_if_exists","convert_xrdf_to_curobo","convert_curobo_to_xrdf"]
