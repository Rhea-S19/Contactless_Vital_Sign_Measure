"""
Measure Respiration Rate

Project : Contactless Vital Sign Meas
"""

import numpy as np
import cv2
import sys

# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid
def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame


def applyFFT(frames, fps):
    n = frames.shape[0]
    t = np.linspace(0, float(n) / fps, n)
    disp = frames.mean(axis=0)
    y = frames - disp

    k = np.arange(n)
    T = n / fps
    frq = k / T  # two sides frequency range
    freqs = frq[range(n // 2)]  # one side frequency range

    Y = np.fft.fft(y, axis=0) / n  # fft computing and normalization
    signals = Y[range(n // 2), :, :]

    return freqs, signals

def bandPass(freqs, signals, freqRange):

    signals[freqs < freqRange[0]] *= 0
    signals[freqs > freqRange[1]] *= 0

    return signals

def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

def freq_from_crossings(sig, fs):
    """Estimate frequency by counting zero crossings

    """
    # print(sig)
    # Find all indices right before a rising-edge zero crossing
    indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
    x = sig[1:]
    x = np.mean(x)

    return x

#from PIL import Image
#from io import BytesIO
#import base64

#def pil_image_to_base64(pil_image):
#    buf = BytesIO()
#    pil_image.save(buf, format="JPEG")
#    return base64.b64encode(buf.getvalue())

#def base64_to_pil_image(base64_img):
#    return Image.open(BytesIO(base64.b64decode(base64_img)))

def searchFreq(freqs, signals, frames, fs):

    curMax = 0
    freMax = 0
    Mi = 0
    Mj = 0
    for i in range(10, signals.shape[1]):
        for j in range(signals.shape[2]):

            idxMax = abs(signals[:, i, j])
            idxMax = np.argmax(idxMax)
            freqMax = freqs[idxMax]
            ampMax = signals[idxMax, i, j]
            c, a = abs(curMax), abs(ampMax)
            if (c < a).any():
                curMax = ampMax
                freMax = freqMax
                Mi = i
                Mj = j
    # print "(%d,%d) -> Freq:%f Amp:%f"%(i,j,freqMax*60, abs(ampMax))

    y = frames[:, Mi, Mj]
    y = y - y.mean()
    fq = freq_from_crossings(y, fs)
    rate_fft = freMax * 60

    rate_count = round(20 + (fq * 10))

    if np.isnan(rate_count):
        rate = rate_fft
    elif abs(rate_fft - rate_count) > 20:
        rate = rate_fft
    else:
        rate = rate_count

    return rate


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
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

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
sampleLen = 10
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels + 1)[levels]
sample = np.zeros((sampleLen, firstGauss.shape[0], firstGauss.shape[1], videoChannels))

idx = 0

respRate = []; resp=[]

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
    if ret == False:
        break

    if len(sys.argv) != 2:
        originalFrame = frame.copy()
        originalVideoWriter.write(originalFrame)

    #detectionFrame = frame[videoHeight//2:realHeight-videoHeight//2, videoWidth//2:realWidth-videoWidth//2, :]

    detectionFrame = frame[int(videoHeight / 2):int(realHeight - videoHeight / 2),int(videoWidth / 2):int(realWidth - int(videoWidth / 2)), :]

    sample[idx] = buildGauss(detectionFrame, levels + 1)[levels]

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

    #print(rr)
    resp.append(rr)
    '''
    if bufferIndex % bpmCalculationFrequency == 0:
        i = i + 1
        for buf in range(bufferSize):
            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
        hz = frequencies[np.argmax(fourierTransformAvg)]
        bpm = 60.0 * hz
        bpmBuffer[bpmBufferIndex] = bpm
        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
    '''
    # Amplify
    #filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
    #filtered = filtered * alpha

    # Reconstruct Resulting Frame
    #filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
    #outputFrame = detectionFrame + filteredFrame
    #outputFrame = cv2.convertScaleAbs(outputFrame)

    #bufferIndex = (bufferIndex + 1) % bufferSize

    '''frame[videoHeight//2:realHeight-videoHeight//2, videoWidth//2:realWidth-videoWidth//2, :] = outputFrame
    cv2.rectangle(frame, (videoWidth//2 , videoHeight//2), (realWidth-videoWidth//2, realHeight-videoHeight//2), boxColor, boxWeight)
    if i > bpmBufferSize:
        cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, fontScale, fontColor, lineType)
        #print("BPM: %d" % bpmBuffer.mean())
        BPM.append(bpmBuffer.mean())
    else:
        cv2.putText(frame, "Calculating BPM...", loadingTextLocation, font, fontScale, fontColor, lineType)

    outputVideoWriter.write(frame)'''

    cv2.putText(frame, "BPM: %d" % rr, bpmTextLocation, font, fontScale, fontColor, lineType)

    if len(sys.argv) != 2:
        cv2.imshow("Webcam Respiration Rate Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
outputVideoWriter.release()
if len(sys.argv) != 2:
    originalVideoWriter.release()

#print(sum(BPM[:20])/20)
rt=sum(resp[40:90])/50
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
