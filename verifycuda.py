from roboflow import Roboflow
rf = Roboflow(api_key="4CPDNdTZEu6FD1ks04AP")
project = rf.workspace("nonbio-trained").project("qp.pet")
version = project.version(2)
dataset = version.download("yolov11")
                