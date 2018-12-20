import os
import sys
import time
from datetime import datetime
import cv2
import openface as of
import pickle
import shutil
import sqlite3

from subprocess import Popen, PIPE

import numpy as np

np.set_printoptions(precision=2)

# openfaceDir = os.path.dirname("/home/frostic/openface/")
clasPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'demos', 'classifier.py')
modelDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
alignedData = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'aligned')
repsData = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'reps')
FASTData = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'FAST')

align = of.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
netof = of.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), 96)

conn = sqlite3.connect('DB')
c = conn.cursor()

Confidence = 0.6  # Accuracy of recognition. If we have accuracy less than this -> new person
Distance = 1      # Distance between faces when we need to learn new person

appearances = []

cam = 0


def getCountOfFaces(numberOfPerson):
    """
    get number of photos of exact person

    :param numberOfPerson: number of person we want to know about
    :return: (int) number of photos
    """
    personData = os.path.join(alignedData, 'person' + str(numberOfPerson + 1))

    lst = os.listdir(personData)

    return len(lst)


def getCountOfPeople():
    """
    get number of persons we know about now

    :return: (int) number of persons
    """
    lst = os.listdir(alignedData)
    res = len(lst)
    for i in range(len(lst)):
        if lst[i].endswith('.t7'):
            res -= 1

    return res


def getReps():
    """
    get representations of first photos of all persons

    :return: (list) list of representations
    """
    lst = os.listdir(alignedData)
    reps = []
    for i in range(len(lst)):
        if not lst[i].endswith('.t7'):
            pathINeed = os.path.join(alignedData, lst[i])
            lst1 = os.listdir(pathINeed)
            alFace = cv2.imread(os.path.join(pathINeed, lst1[0]))
            reps.append(netof.forward(alFace))
    return reps


def infer(rep):
    """
    ask our programm, who is it

    :param rep: representation of face we need to recognize
    :return: name of person or 'NO' if we dont sure
    """
    classifierPath = os.path.join(repsData, 'classifier.pkl')
    with open(classifierPath, 'rb') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
        else:
                (le, clf) = pickle.load(f, encoding='latin1')
    rep = rep.reshape(1, -1)
    predictions = clf.predict_proba(rep).ravel()
    maxI = np.argmax(predictions)
    person = le.inverse_transform(maxI)
    confidence = predictions[maxI]

    print confidence, person

    if confidence > Confidence:
        return person

    return 'NO'


def clear_non_trainable():
    """
    clear directories with not enough photos to train on them

    :return: numbers of cleared diretories
    """
    cleared = []
    lst = os.listdir(alignedData)

    for i in range(len(lst)):
        if not lst[i].endswith('.t7'):
            numberOfFaces = len(os.listdir(os.path.join(alignedData, lst[i])))

            if numberOfFaces < 6:
                shutil.move(os.path.join(alignedData, lst[i]),
                            os.path.join(FASTData, datetime.strftime(datetime.now(), "%Y.%m.%d-%H:%M:%S" + '-' + str(i))))
                cleared.append(lst[i])
    return cleared


def pretrain():
    """
    prepare everything to train on new persons (make batch of photos)

    :return: nothing
    """
    peopleCount = getCountOfPeople()
    reps = getReps()
    lst = os.listdir(alignedData)

    for i in range(15):
        capture = cv2.VideoCapture(cam)
        ret, frame = capture.read()
        cv2.waitKey(10)
        capture.release()
        rgbimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bb = align.getAllFaceBoundingBoxes(rgbimg)
        for j in range(len(bb)):
            alignedface = align.align(imgDim=96, rgbImg=rgbimg, bb=bb[j],
                                      landmarkIndices=of.AlignDlib.OUTER_EYES_AND_NOSE)
            if not (alignedface is None):
                d = []
                rep1 = netof.forward(alignedface)
                for k in range(peopleCount):
                    a = np.dot(reps[k] - rep1, reps[k] - rep1)
                    d.append(a)
                minimum = 0
                for k in range(peopleCount):
                    if d[minimum] > d[k]:
                        minimum = k
                print('minimum ' + str(d[minimum]) + ': ' + str(minimum))
                if d[minimum] < Distance:
                    pathINeed = os.path.join(alignedData, lst[minimum])
                    bgrimg = cv2.cvtColor(alignedface, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(pathINeed, lst[minimum] + '-' + str(i + 2) + '.png'),
                                bgrimg)


def aligning():
    """
    highlight faces on photos that were added when prog was down

    :return: nothing
    """
    directoryName = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'photos')
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'util', 'align-dlib.py'),
           directoryName, 'align', 'outerEyesAndNose',
           alignedData]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert p.returncode == 0


def train():
    """
    train classifier on new persons
    :return: nothing (new classifier saved like file)
    """
    start = time.time()

    cmd = [os.path.join(os.path.dirname(os.path.realpath(__file__)), 'batch-represent', 'main.lua'),
           '-outDir', repsData, '-data', alignedData]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert p.returncode == 0

    cmd = [sys.executable, clasPath, 'train', repsData]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)
    assert p.returncode == 0

    print("Training took {} seconds.".format(time.time() - start))
    os.remove(os.path.join(alignedData, 'cache.t7'))


def do_everything(num):
    """
    main module of my program, processes photos and sticks everything together

    :param num: number of photos to be processed
    :return: (list) list of persons who were on photo
    """
    f = open("log.txt", "a")
    for i in range(num):
        apps = []
        capture = cv2.VideoCapture(cam)
        ret, frame = capture.read()
        cv2.waitKey(10)
        cv2.imwrite('data.jpg', frame)
        capture.release()

        bgrimg = cv2.imread('data.jpg')
        rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)
        cv2.waitKey(10)
        bb = align.getAllFaceBoundingBoxes(rgbimg)
        if bb is None:
            print 'bb is None'
            continue

        print('Found {} faces.'.format(len(bb)))

        peopleCount1 = getCountOfPeople()

        for j in range(len(bb)):
            alignedface = align.align(imgDim=96, rgbImg=rgbimg, bb=bb[j],
                                      landmarkIndices=of.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedface is None:
                print 'AlignedFace is None'
                continue

            bgrimg = cv2.cvtColor(alignedface, cv2.COLOR_RGB2BGR)
            cv2.imwrite('face.jpg', bgrimg)

            peopleCount = getCountOfPeople()

            if peopleCount > 1:
                name = infer(netof.forward(alignedface))
                if name == 'NO':
                    os.mkdir(os.path.join(alignedData, 'person' + str(peopleCount + 1)))
                    cv2.imwrite(os.path.join(alignedData, 'person' + str(peopleCount + 1),
                                             'person' + str(peopleCount + 1) + '-' + str(1) + '.png'), bgrimg)
                    f.write(datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S") + ': '
                            + 'person' + str(peopleCount + 1) + ' was on camera\n')

                    c.execute("INSERT INTO personid VALUES ('%s', NULL);" % (peopleCount + 1))
                    conn.commit()

                    apps.append('person' + str(peopleCount + 1))
                    print(datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S") + ': '
                          + 'person' + str(peopleCount + 1) + ' was on camera')

                else:
                    if name.find('person') != -1:
                        c.execute("SELECT * FROM personid WHERE id='%s'" % name[6])
                        conn.commit()
                        results = c.fetchall()
                        if len(results) != 0:
                            if results[0][1] is not None:
                                namepers = results[0][1]
                            else:
                                namepers = name
                        else:
                            namepers = name
                    else:
                        namepers = name
                    if appearances.count(name) == 0:
                        f.write(datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S") + ': '
                                + namepers + ' was on camera\n')
                    apps.append(namepers)
                    print(datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S") + ': '
                          + namepers + ' was on camera')
            else:
                if peopleCount == 1:
                    rep1 = netof.forward(alignedface)
                    reps = getReps()
                    d = np.dot(reps[0] - rep1, reps[0] - rep1)
                    if d < Distance:
                        c.execute("SELECT * FROM personid WHERE id='%s'" % 1)
                        conn.commit()
                        results = c.fetchall()
                        print results
                        if len(results) != 0:
                            if results[0][1] is not None:
                                namepers = results[0][1]
                            else:
                                namepers = 'person1'
                        else:
                            namepers = 'person1'
                        if appearances.count('person' + str(1)) == 0:
                            f.write(datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S") + ': '
                                    + namepers + ' was on camera\n')

                        apps.append(namepers)
                        print(datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S") + ': '
                              + namepers + ' was on camera')
                        continue

                os.mkdir(os.path.join(alignedData, 'person' + str(peopleCount + 1)))
                cv2.imwrite(os.path.join(alignedData, 'person' + str(peopleCount + 1),
                                         'person' + str(peopleCount + 1) + '-' + str(1) + '.png'), bgrimg)

                if appearances.count('person' + str(peopleCount + 1)) == 0:
                    f.write(datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S") + ': '
                            + 'person' + str(peopleCount + 1) + ' was on camera\n')
                    c.execute("INSERT INTO personid VALUES ('%s', NULL);" % (peopleCount + 1))
                    conn.commit()

                apps.append('person' + str(peopleCount + 1))
                print(datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S") + ': '
                      + 'person' + str(peopleCount + 1) + ' was on camera')

        if getCountOfPeople() - peopleCount1 > 0:
            pretrain()
            cleared1 = clear_non_trainable()
            for ki in range(len(cleared1)):
                f.write('\t' + str(cleared1[ki]) + ' was on camera, but we couldnt get enough photos of him/her.' +
                        'Classifier was not trained for him\n')
                print cleared1[ki]
                c.execute("DELETE FROM personid WHERE id = '%s'" % cleared1[ki][6:])
                conn.commit()

        if ((getCountOfPeople() - peopleCount1) > 0 and peopleCount1 != 0) or \
                ((getCountOfPeople() - peopleCount1) > 1 and peopleCount1 == 0):
            train()
        for _ in range(len(appearances)):
            appearances.pop()
        for _ in range(len(apps)):
            appearances.append(apps.pop())
        print appearances

    f.close()
    return appearances
