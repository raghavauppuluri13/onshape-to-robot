#!/usr/bin/env python3

import sys
import os
import subprocess
from pathlib import Path

# Script taken from doing the needed operation
# (Filters > Remeshing, Simplification and Reconstruction >
# Quadric Edge Collapse Decimation, with parameters:
# 0.9 percentage reduction (10%), 0.3 Quality threshold (70%)
# Target number of faces is ignored with those parameters
# conserving face normals, planar simplification and
# post-simplimfication cleaning)
# And going to Filter > Show current filter script
filter_script_mlx = """<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Simplification: Clustering Decimation">
  <Param name="Threshold" type="RichAbsPerc" max="0.247701" isxmlparam="0" description="Cell Size" tooltip="The size of the cell of the clustering grid. Smaller the cell finer the resulting mesh. For obtaining a very coarse mesh use larger values." min="0" value="0.002477"/>
  <Param name="Selected" type="RichBool" isxmlparam="0" description="Affect only selected faces" tooltip="If selected the filter affect only the selected faces" value="false"/>
 </filter>
</FilterScript>
"""


def reduce_faces(in_file, filter_script_path=Path(__file__).parent / "simplify_mesh_filter.mlx"):
    in_file = Path(in_file)
    # Add input mesh
    command = f"meshlabserver -i {str(in_file.absolute())}"
    # Add the filter script
    command += f" -s {str(filter_script_path.absolute())}"
    # Add the output filename and output flags
    command += f" -o {str(in_file.absolute())}"
    # Execute command
    os.system(command)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage:")
        print(f"{sys.argv[0]} /path/to/input_mesh num_iterations")
        print("For example, reduce 10 times:")
        print(f"{sys.argv[0]} /home/myuser/mymesh.stl 10")
        exit(0)

    in_mesh = Path(sys.argv[1])
    num_iterations = int(sys.argv[2])

    print(f"Input mesh: {in_mesh} (filename: {in_mesh.name})")

    print()
    print(f"Done reducing, the file is at: {in_mesh}")
