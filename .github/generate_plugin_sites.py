import glob
from jinja2 import Template
import yaml
import os
import json

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


pluginList = []
ymlFiles = glob.glob(r"../plugins_repo/*.yml")
ymlFiles.sort(key=alphaSort)
for ymlFile in ymlFiles:
    try:
        with open(ymlFile, "r") as stream:
            pluginObject = yaml.safe_load(stream)
        pluginObject["yml"] = os.path.basename(ymlFile)
        pyFile = ymlFile.replace(".yml", ".py")
        if os.path.isfile(pyFile):
            pluginObject["py"] = os.path.basename(pyFile)
        jsFile = ymlFile.replace(".yml", ".js")
        if os.path.isfile(jsFile):
            pluginObject["js"] = os.path.basename(jsFile)
        pluginList.append(pluginObject)
    except:
        pass

with open("../plugins_repo/pluginList.json", "w") as f:
    json.dump(pluginList, f, indent=4, sort_keys=True, default=str)

with open("../plugins_repo/index.html", "w") as f:
    f.write(t.render(pluginList=pluginList))
