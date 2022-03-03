import glob
from jinja2 import Template
import os
import json

t = Template("""
<h1>TissUUmaps plugins</h1>
{% for plugin in pluginList %}\
    <h2>{{plugin.name}}</h2>
    {% if plugin.img %}\
        <img src="{{plugin.img}}" width="500px" style="max-height:500px;"/><br/>
    {% endif %}
    {% if plugin.txt %}\
        <b>Description:</b> {{plugin.txt}}
    {% endif %}
{% endfor %}\
<table>
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