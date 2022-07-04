"""
Measure SpO2 Oxygen Level

Project : Contactless Vital Sign Meas
"""

import numpy as np
import cv2
import sys

# Helper Methods
#def

# Webcam Parameters
webcam = None
if len(sys.argv) == 2:
    webcam = cv2.VideoCapture(sys.argv[1])
else:
    webcam = cv2.VideoCapture(0)
realWidth = 320
realHeight = 240
videoWidth = 160
videoHeight = 120
videoChannels = 3
videoFrameRate = 15
webcam.set(3, realWidth)
webcam.set(4, realHeight)

# Output Videos
if len(sys.argv) != 2:
    originalVideoFilename = "original.mov"
    originalVideoWriter = cv2.VideoWriter()
    originalVideoWriter.open(originalVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate, (realWidth, realHeight), True)

outputVideoFilename = "output.mov"
outputVideoWriter = cv2.VideoWriter()
outputVideoWriter.open(outputVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate, (realWidth, realHeight), True)

# Color Magnification Parameters
count = 0
i=0
A=100
B=5
bo = 0.0

spresult = 0
spcount=0
result= 0

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth//2 + 5, 30)
fontScale = 1
fontColor = (255,255,255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

#Initial Gaussian Pyramid
#sampleLen = 10
#firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
#firstGauss = buildGauss(firstFrame, levels + 1)[levels]
#sample = np.zeros((sampleLen, firstGauss.shape[0], firstGauss.shape[1], videoChannels))

#idx = 0

#respRate = []; resp=[]

# pipeline = PipeLine(videoFrameRate)
#face_flag = 0
#for i in range(len(videoStrings)):
    #input_img = base64_to_pil_image(videoStrings[i])

    #input_img = input_img.resize((320, 240))
    #gray = cv2.cvtColor(np.array(input_img), cv2.COLOR_BGR2GRAY)

    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # print(faces)
    # faces = [1,2,3]
    # print(len(faces))
    #if len(faces) > 0:
        # print("FACE FOUND _ RR")

        #face_flag = 1

        #frame = cv2.cvtColor(np.array(input_img), cv2.COLOR_BGR2RGB)

while (True):
    ret, frame = webcam.read()
    #print(frame)
    if ret == False:
        break

    if len(sys.argv) != 2:
        originalFrame = frame.copy()
        originalVideoWriter.write(originalFrame)

    #detectionFrame = frame[videoHeight//2:realHeight-videoHeight//2, videoWidth//2:realWidth-videoWidth//2, :]

    #detectionFrame = frame[int(videoHeight / 2):int(realHeight - videoHeight / 2),int(videoWidth / 2):int(realWidth - int(videoWidth / 2)), :]

    # Red channel operations
    red_channel = frame[:, :, 2]
    mean_red = np.mean(red_channel)
    # print("RED MEAN", mean_red)
    std_red = np.std(red_channel)
    # print("RED STD", std_red)
    red_final = std_red / mean_red
    # print("RED FINAL",red_final)

    # Blue channel operations
    blue_channel = frame[:, :, 0]
    mean_blue = np.mean(blue_channel)
    # print("BLUE MEAN", mean_blue)
    std_blue = np.std(red_channel)
    # print("BLUE STD", std_blue)
    blue_final = std_blue / mean_blue
    # print("BLUE FINAL",blue_final)

    sp = A - (B * (red_final / blue_final))
    sp = round(sp, 2)
    spresult = spresult + sp +3 #(correction factor)
    spcount += 1

    spmean=float(spresult)/float(spcount)
    '''sample[idx] = buildGauss(detectionFrame, levels + 1)[levels]

    freqs, signals = applyFFT(sample, videoFrameRate)
    signals = bandPass(freqs, signals, (0.2, 0.8))
    respiratoryRate = searchFreq(freqs, signals, sample, videoFrameRate)

    # frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):(realWidth-int(videoWidth/2)), :] = outputFrame

    idx = (idx + 1) % 10

    respRate.append(respiratoryRate)

    #else:
        #print("Face not found")

    #if face_flag == 1:
    l = []
    a = max(respRate)
    b = np.mean(respRate)
    if b < 0:
        b = 5
    l.append(a)
    l.append(b)

    rr = np.mean(l)
    rr = round(rr, 2)
    #else:
     #   rr = "Face not recognised!"

    print(rr)
    resp.append(rr)
    
    if bufferIndex % bpmCalculationFrequency == 0:
        i = i + 1
        for buf in range(bufferSize):
            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
        hz = frequencies[np.argmax(fourierTransformAvg)]
        bpm = 60.0 * hz
        bpmBuffer[bpmBufferIndex] = bpm
        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
    
    # Amplify
    #filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
    #filtered = filtered * alpha

    # Reconstruct Resulting Frame
    #filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
    #outputFrame = detectionFrame + filteredFrame
    #outputFrame = cv2.convertScaleAbs(outputFrame)

    #bufferIndex = (bufferIndex + 1) % bufferSize
    '''
    #frame[videoHeight//2:realHeight-videoHeight//2, videoWidth//2:realWidth-videoWidth//2, :] = outputFrame
    #cv2.rectangle(frame, (videoWidth//2 , videoHeight//2), (realWidth-videoWidth//2, realHeight-videoHeight//2), boxColor, boxWeight)
    #if i > bpmBufferSize:
    cv2.putText(frame, "SpO2: %d" % spmean, bpmTextLocation, font, fontScale, fontColor, lineType)
    #print("BPM: %d" % bpmBuffer.mean())
    #BPM.append(sp)
    #else:
     #   cv2.putText(frame, "Calculating spO2...", loadingTextLocation, font, fontScale, fontColor, lineType)

    outputVideoWriter.write(frame)

    if len(sys.argv) != 2:
        cv2.imshow("Webcam Heart Rate Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
outputVideoWriter.release()
if len(sys.argv) != 2:
    originalVideoWriter.release()

res=1
result = result + res
print(sp)
#video_strings = frame.split(' ')
#video_strings = video_strings[1:]
#print(video_strings[0]==video_strings[4])
#video_strings = video_strings*2

length = len(frame[1:])
print("length" +str(length))
#result = result/length
result = spresult/spcount
print("final res value: "+ str(result))

if result > 0.25:
    spresult = spresult / spcount
    spresult = round(spresult, 2)
else:
    spresult = "Finger not recognised"


#print(sum(BPM[:20])/20)
'''spO=sum(sp[20:50])/30
rt=round(rt,2)
import json

# a Python object (dict):
x = {
  "Respiration Rate": rt
}

# convert into JSON:
y = json.dumps(x)

# the result is a JSON string:
print(y)
'''