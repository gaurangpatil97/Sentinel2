# from roboflow import Roboflow
# rf = Roboflow(api_key="gj1dDXzQnsRNBuYJCmoJ")
# project = rf.workspace("gaurang-patil-gr0h9").project("helmet-detection-ntbfz-yozkl")
# version = project.version(1)
# dataset = version.download("yolov8", location = r"C:\Users\gaura\Desktop\Sentinel\models\HelmetDetection")
                


from roboflow import Roboflow

try:
    rf = Roboflow(api_key="gj1dDXzQnsRNBuYJCmoJ")
    workspace = rf.workspace("gaurang-patil-gr0h9")
    project = workspace.project("helmet-detection-ntbfz-yozkl")
    version = project.version(1)
    
    dataset = version.download(
        "yolov8", 
        location=r"C:\Users\gaura\Desktop\Sentinel\models\HelmetDetection\dataset"
    )
    print("Download complete!", dataset.location)

except Exception as e:
    print("ERROR:", e)