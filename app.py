#Contacless Vital Sign Measurement

from flask import Flask, render_template, Response
import cv2
import numpy as np
import sys

app=Flask(__name__)
webcam = cv2.VideoCapture(0)

# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid
def reconstructFrame(pyramid, index, levels):
    videoWidth = 160
    videoHeight = 120
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



def gen_frames():
    # Webcam Parameters
    '''webcam = None
    if len(sys.argv) == 2:
        webcam = cv2.VideoCapture(sys.argv[1])
    else:
        webcam = cv2.VideoCapture(0)'''
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
        originalVideoWriter.open(originalVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate,
                                 (realWidth, realHeight), True)

    outputVideoFilename = "output.mov"
    outputVideoWriter = cv2.VideoWriter()
    outputVideoWriter.open(outputVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate,
                           (realWidth, realHeight), True)

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
    bpmTextLocation = (videoWidth // 2 + 5, 30)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    boxColor = (0, 255, 0)
    boxWeight = 3

    # Initialize Gaussian Pyramid
    firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
    firstGauss = buildGauss(firstFrame, levels + 1)[levels]
    videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
    fourierTransformAvg = np.zeros((bufferSize))

    # Bandpass Filter for Specified Frequencies
    frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
    mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

    # Heart Rate Calculation Variables
    bpmCalculationFrequency = 15
    bpmBufferIndex = 0
    bpmBufferSize = 10
    bpmBuffer = np.zeros((bpmBufferSize))

    i = 0;
    BPM = [];

    while True:
        success, frame = webcam.read()  # read the camera frame
        if not success:
            break
        else:
            detectionFrame = frame[videoHeight // 2:realHeight - videoHeight // 2,videoWidth // 2:realWidth - videoWidth // 2, :]

            # Construct Gaussian Pyramid
            videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
            fourierTransform = np.fft.fft(videoGauss, axis=0)

            # Bandpass Filter
            fourierTransform[mask == False] = 0

            # Grab a Pulse
            if bufferIndex % bpmCalculationFrequency == 0:
                i = i + 1
                for buf in range(bufferSize):
                    fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                hz = frequencies[np.argmax(fourierTransformAvg)]
                bpm = 60.0 * hz
                bpmBuffer[bpmBufferIndex] = bpm
                bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

            # Amplify
            filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
            filtered = filtered * alpha

            # Reconstruct Resulting Frame
            filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
            outputFrame = detectionFrame + filteredFrame
            outputFrame = cv2.convertScaleAbs(outputFrame)

            bufferIndex = (bufferIndex + 1) % bufferSize

            frame[videoHeight // 2:realHeight - videoHeight // 2, videoWidth // 2:realWidth - videoWidth // 2,:] = outputFrame
            cv2.rectangle(frame, (videoWidth // 2, videoHeight // 2),(realWidth - videoWidth // 2, realHeight - videoHeight // 2), boxColor, boxWeight)
            if i > bpmBufferSize:
                cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, fontScale, fontColor, lineType)
                # print("BPM: %d" % bpmBuffer.mean())
                BPM.append(bpmBuffer.mean())
            else:
                cv2.putText(frame, "Calculating BPM...", loadingTextLocation, font, fontScale, fontColor, lineType)

            outputVideoWriter.write(frame)

            if len(sys.argv) != 2:
                cv2.imshow("Webcam Heart Rate Monitor", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            #webcam.release()
            #cv2.destroyAllWindows()
            #outputVideoWriter.release()
            '''
            detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            faces=detector.detectMultiScale(frame,1.1,7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             #Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            '''
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_resp():
    # Webcam Parameters
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
        originalVideoWriter.open(originalVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate,
                                 (realWidth, realHeight), True)

    outputVideoFilename = "output.mov"
    outputVideoWriter = cv2.VideoWriter()
    outputVideoWriter.open(outputVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate,
                           (realWidth, realHeight), True)

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
    bpmTextLocation = (videoWidth // 2 + 5, 30)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    boxColor = (0, 255, 0)
    boxWeight = 3

    # Initial Gaussian Pyramid
    sampleLen = 10
    firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
    firstGauss = buildGauss(firstFrame, levels + 1)[levels]
    sample = np.zeros((sampleLen, firstGauss.shape[0], firstGauss.shape[1], videoChannels))

    idx = 0

    respRate = [];
    resp = []

    while True:
        success, frame = webcam.read()  # read the camera frame
        if not success:
            break
        if len(sys.argv) != 2:
            originalFrame = frame.copy()
            originalVideoWriter.write(originalFrame)

        # detectionFrame = frame[videoHeight//2:realHeight-videoHeight//2, videoWidth//2:realWidth-videoWidth//2, :]

        detectionFrame = frame[int(videoHeight / 2):int(realHeight - videoHeight / 2),
                         int(videoWidth / 2):int(realWidth - int(videoWidth / 2)), :]

        sample[idx] = buildGauss(detectionFrame, levels + 1)[levels]

        freqs, signals = applyFFT(sample, videoFrameRate)
        signals = bandPass(freqs, signals, (0.2, 0.8))
        respiratoryRate = searchFreq(freqs, signals, sample, videoFrameRate)

        # frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):(realWidth-int(videoWidth/2)), :] = outputFrame

        idx = (idx + 1) % 10

        respRate.append(respiratoryRate)

        # else:
        # print("Face not found")

        # if face_flag == 1:
        l = []
        a = max(respRate)
        b = np.mean(respRate)
        if b < 0:
            b = 5
        l.append(a)
        l.append(b)

        rr = np.mean(l)
        rr = round(rr, 2)
        # else:
        #   rr = "Face not recognised!"

        # print(rr)
        resp.append(rr)

        cv2.putText(frame, "BPM: %d" % rr, bpmTextLocation, font, fontScale, fontColor, lineType)

        if len(sys.argv) != 2:
            cv2.imshow("Webcam Respiration Rate Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        '''
        frame[videoHeight // 2:realHeight - videoHeight // 2, videoWidth // 2:realWidth - videoWidth // 2,:] = outputFrame
        cv2.rectangle(frame, (videoWidth // 2, videoHeight // 2),(realWidth - videoWidth // 2, realHeight - videoHeight // 2), boxColor, boxWeight)
        if i > bpmBufferSize:
            cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, fontScale, fontColor, lineType)
            # print("BPM: %d" % bpmBuffer.mean())
            BPM.append(bpmBuffer.mean())
        else:
            cv2.putText(frame, "Calculating BPM...", loadingTextLocation, font, fontScale, fontColor, lineType)

        outputVideoWriter.write(frame)

        if len(sys.argv) != 2:
            cv2.imshow("Respiration Rate Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        '''
        #webcam.release()
        #cv2.destroyAllWindows()
        #outputVideoWriter.release()
        '''
        detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
        faces=detector.detectMultiScale(frame,1.1,7)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         #Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        '''
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_spo2():
    # Webcam Parameters
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
        originalVideoWriter.open(originalVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate,
                                 (realWidth, realHeight), True)

    outputVideoFilename = "output.mov"
    outputVideoWriter = cv2.VideoWriter()
    outputVideoWriter.open(outputVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate,
                           (realWidth, realHeight), True)

    #Calculation Parameters
    count = 0
    i = 0
    A = 100
    B = 5
    bo = 0.0

    spresult = 0
    spcount = 0
    result = 0

    # Output Display Parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    loadingTextLocation = (20, 30)
    bpmTextLocation = (videoWidth // 2 + 5, 30)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    boxColor = (0, 255, 0)
    boxWeight = 3

    while True:
        success, frame = webcam.read()  # read the camera frame
        if not success:
            break
        if len(sys.argv) != 2:
            originalFrame = frame.copy()
            originalVideoWriter.write(originalFrame)

        # detectionFrame = frame[videoHeight//2:realHeight-videoHeight//2, videoWidth//2:realWidth-videoWidth//2, :]

        # detectionFrame = frame[int(videoHeight / 2):int(realHeight - videoHeight / 2),int(videoWidth / 2):int(realWidth - int(videoWidth / 2)), :]

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
        spresult = spresult + sp
        spcount += 1

        spmean = float(spresult) / float(spcount)

        cv2.putText(frame, "SpO2: %d" % spmean, bpmTextLocation, font, fontScale, fontColor, lineType)

        if len(sys.argv) != 2:
            cv2.imshow("Webcam SpO2 Level Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        #webcam.release()
        #cv2.destroyAllWindows()
        #outputVideoWriter.release()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/resp_feed')
def resp_feed():
    return Response(gen_resp(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/spo2_feed')
def spo2_feed():
    return Response(gen_spo2(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)