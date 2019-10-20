# DetectCarParkingSlot_Contest
QualifiedRound
    Contains qualified round proposed solution. The accuracy is hardly 10-20% using Yolo model prre-trained model.

FinalRound_ImprovedAccuracy_Functionality
    It detects car and empty slots in car park area with 90% accuracy. Used Yolo v3 model and PUCPR dataset to train Object detection model for Car parking area. Using trained model, collected all parking slots for Fully car parked image and similarly predict occupied car slot for interference. IOU (intersection over union) and NMS(non maximum suppression) technique used to find empty slots for inference image.

