# APMI-Tools

## Export a mesh for the character screen and the field

1. Capture the same unmodified character in the character screen and in the field with XXMI Frame Analysis. Keep both capture folders.
2. Import the character screen's `log.txt` and edit the mesh in Blender.
3. At export, enable **Export Full Mesh** and set **Field Frame Analysis** to the field capture folder. Install the export in an APMI package with Core 0.3.1 or newer.

The exporter compares the original vertex IDs and bone weights to map the edited mesh to each captured bone palette. It can also use a verified bone from another component on the same actor. The game draws the original component if a required bone, matching skinning input, or actor snapshot is unavailable. Check the export warnings for parts that lack a verified field mapping.

Leave **Export Full Mesh** off for mesh deletions that retain the original vertex data.
