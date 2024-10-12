from . import nodes

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadDepthPro": nodes.LoadDepthPro,
    "RunDepthPro": nodes.RunDepthPro
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDepthPro": "DownLoad And Load DepthPro",
    "RunDepthPro": "Run DepthPro",
}