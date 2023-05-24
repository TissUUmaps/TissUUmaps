import glob
import json
import os
import shutil

import yaml
from jinja2 import Template

t = Template(
    """
<html>
    <head>
        <title>TissUUmaps plugins</title>
        <style>
            table {
                color: #333; /* Lighten up font color */
                font-family: Helvetica, Arial, sans-serif; /* Nicer font */
                width: 100%;
                border-collapse:
                collapse; border-spacing: 0;
                max-width:1600px;
                margin-left:auto;
                margin-right:auto;
            }

            td, th { border: 1px solid #CCC; height: 30px; } /* Make cells a bit taller */

            th {
                background: #F3F3F3; /* Light grey background */
                font-weight: bold; /* Make sure they're bold */
            }

            td {
                background: #FAFAFA; /* Lighter grey background */
                text-align: left; /* Center our text */
                vertical-align: top;
                padding: 10px;
            }
            td.thumb {
                width: 33%;
                white-space: nowrap;
            }
        </style>
    </head>
    <body>
        <h1>TissUUmaps plugins</h1>
        <table>
        {% for plugin in pluginList %}\
                <tr>
                    <td>
                        <h2>{{plugin.name}}:</h2>
                        {% if plugin.author %}\
                            <p><i>Author: {{plugin.author}} </i></p>
                        {% endif %}
                        {% if plugin.version %}\
                            <p><i>Version: {{plugin.version}}</i></p>
                        {% endif %}
                        {% if plugin.description %}\
                            <h3>Description</h3>
                            {{plugin.description}}
                        {% endif %}
                        <h3>Plugin Files</h3>
                        <ul>
                            <li><a href="{{plugin.yml}}">{{plugin.yml}}</a></li>
                            {% if plugin.py %}
                                <li><a href="{{plugin.py}}">{{plugin.py}}</a></li>
                            {% endif %}
                            {% if plugin.js %}
                                <li><a href="{{plugin.js}}">{{plugin.js}}</a></li>
                            {% endif %}
                        </ul>
                    </td>
                    <td class="thumb">
                        {% if plugin.image %}\
                            <img src="{{plugin.image}}" style="max-height:500px;width:100%;"/><br/>
                        {% endif %}
                    </td>
                </tr>
        {% endfor %}\
        </table>
        <h2>Files needed by TissUUmaps for plugin installation:</h2>
        <ul><li><a href="pluginList.json">pluginList.json</a></li></ul>
        <table>
    </body>
</html>
"""
)


def alphaSort(key):
    if "Plugin_template" in key:
        return "ZZZ"
    return key


version_file = os.path.join("..", "tissuumaps", "VERSION")

with open(version_file) as f:
    tissuumaps_version = f.read().strip()[:3]

old_versions = [
    os.path.basename(dir)
    for dir in glob.glob(r"../plugins_repo/*")
    if os.path.isdir(dir)
]


def getPlugin(ymlFile):
    with open(ymlFile, "r") as stream:
        pluginObject = yaml.safe_load(stream)
    pluginObject["yml"] = os.path.basename(ymlFile)
    pyFile = ymlFile.replace(".yml", ".py")
    if os.path.isfile(pyFile):
        pluginObject["py"] = os.path.basename(pyFile)
    jsFile = ymlFile.replace(".yml", ".js")
    if os.path.isfile(jsFile):
        pluginObject["js"] = os.path.basename(jsFile)
    if "image" in pluginObject.keys():
        pluginObject["image"] = pluginObject["image"]
    return pluginObject


# For compatibility reasons, we keep v3.0 in the main folder:
v3_0Files = glob.glob(rf"../plugins_repo/3.0/*")
for src in v3_0Files:
    shutil.copyfile(src, src.replace("plugins_repo/3.0", "plugins_repo/"))

for version in ["."] + old_versions:
    pluginList = []

    ymlFiles = glob.glob(rf"../plugins_repo/{version}/*.yml")
    ymlFiles.sort(key=alphaSort)
    for ymlFile in ymlFiles:
        try:
            pluginList.append(getPlugin(ymlFile))
        except:
            pass

    if version != ".":
        with open(f"../plugins_repo/{version}/index.html", "w") as f:
            f.write(
                t.render(pluginList=pluginList, tissuumaps_version=tissuumaps_version)
            )
    with open(f"../plugins_repo/{version}/pluginList.json", "w") as f:
        json.dump(pluginList, f, indent=4, sort_keys=True, default=str)
