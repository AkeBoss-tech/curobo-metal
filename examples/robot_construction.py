"""Load a packaged robot through the pinned public builder/parser APIs."""

from curobo.content import get_assets_path
from curobo.robot_builder import RobotBuilder
from curobo.robot_parser import UrdfRobotParser


urdf = get_assets_path() / "robot/franka_description/franka_panda.urdf"
parser = UrdfRobotParser(str(urdf))
builder = RobotBuilder(str(urdf), tool_frames=["panda_hand"])
print(parser.root_link, len(parser.get_link_names()), builder.build().tool_frames)
