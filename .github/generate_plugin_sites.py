import glob
from jinja2 import Template
import os
import json

t = Template("""
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
                width: 20%;
                white-space: nowrap;
            }
        </style>
    </head>
    <body>
        <h1>TissUUmaps plugins</h1>
        {% for plugin in pluginList %}\
            <h2>{{plugin.name}}:</h2>
            <table>
                <tr>
                    {% if plugin.img %}\
                        <td class="thumb">
                            <img src="{{plugin.img}}" style="max-height:500px;width:100%;"/><br/>
                        </td>
                    {% endif %}
                    <td>
                        {% if plugin.txt %}\
                            <h3>Description</h3>
                            {{plugin.txt}}
                        {% endif %}
                        <h3>Plugin Files</h3>
                        <ul>
                            {% if plugin.py %}
                                <li><a href="{{plugin.py}}">{{plugin.py}}</a></li>
                            {% endif %}
                            {% if plugin.js %}
                                <li><a href="{{plugin.js}}">{{plugin.js}}</a></li>
                            {% endif %}
                        </ul>
                    </td>
                </tr>
            </table>
            <hr/>
        {% endfor %}\
        <h2>Files needed by TissUUmaps for plugin installation:</h2>
        <ul><li><a href="pluginList.json">pluginList.json</a></li></ul>
        <table>
    </body>
</html>
""")

pluginList = []
for pythonFile in glob.glob(r"../tissuumaps/plugins_available/*.py"):
    print (pythonFile)
    pluginObject = {
        "py":os.path.basename(pythonFile), 
        "name":os.path.basename(pythonFile).replace(".py","").replace("_"," ")
    }
    
    for imgFmt in [".png",".gif",".jpg",".jpeg"]:
        imgFile = pythonFile.replace(".py",imgFmt)
        if os.path.isfile(pythonFile.replace(".py",imgFmt)):
            pluginObject["img"] = os.path.basename(imgFile)
    txtFile = pythonFile.replace(".py",".txt")
    if os.path.isfile(txtFile):
        with open(txtFile) as f:
            pluginObject["txt"] = f.read()
    jsFile = pythonFile.replace(".py",".js")
    if os.path.isfile(jsFile):
        pluginObject["js"] = os.path.basename(jsFile)
    pluginList.append(pluginObject)

with open("../tissuumaps/plugins_available/pluginList.json","w") as f:
    json.dump(pluginList, f, indent=4)

with open("../tissuumaps/plugins_available/index.html","w") as f:
    f.write(t.render(pluginList=pluginList))