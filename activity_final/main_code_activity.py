import numpy as np
import cv2
import imutils 
import time
from datetime import datetime
from CNN_predict import detect_activities 
from PIL import Image
from maxPool import connectedLayers
# Initializing the HOG person 
# detector 

hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
def norm_pdf(x,mean,sigma):
    return (1/(np.sqrt(2*3.14)*sigma))*(np.exp(-0.5*(((x-mean)/sigma)**2)))

def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)
def CNN_test(train_model,test,flatten):
    if 0:
        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.layers import Flatten
        from keras.layers import Dense
        
        # Initialising the CNN
        classifier = Sequential()
        
        # Step 1 - Convolution
        # Convolution - input image, applying feature detectors => feature map
        # 3D Array because colored images
        classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
        
        # Step 2 - Pooling
        # Feature Map - Take Max -> Pooled Feature Map, reduced size, reduce complexity
        # without losing performance, don't lose spatial structure
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding second convolution layer
        # don't need to include input_shape since we're done it
        classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Step 3 - Flattening
        # Pooled Feature Maps apply flattening maps to a huge vector 
        # for a future ANN that is fully-conntected
        # Why don't we lose spatial structure by flattening?
        # We don't because the high numbers from convolution feature from the feature detector
        # Max Pooling keeps them these high numbers, and flattening keeps these high numbers
        # Why didn't we take all the pixels and flatten into a huge vector?
        # Only pixels of itself, but not how they're spatially structured around it
        # But if we apply convolution and pooling, since feature map corresponds to each feature 
        # of an image, specific image unique pixels, we keep the spatial structure of the picture.
        classifier.add(Flatten())
        
        
        # Step 4 - Full Connection
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = 1, activation = 'sigmoid'))
        
        # Compile - SGD, Loss Function, Performance Metric
        # Logarithmic loss - binary cross entropy, more than two outcomes, categorical cross entropy
        # Metrics is the accuracy metric
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        # part 2 - Fitting the CNN to the images 
        # Keras preprocessing images to prevent overfitting, image augmentation, 
        # great accuracy on training poor results on test sets
        # Need lots of images to find correlations, patterns in pixels
        # Find patterns in pixels, 10000 images, 8000 training, not much exactly or use a trick
        # Image augmentation will create batches and each batch will create random transformation
        # leading to more diverse images and more training
        # Image augmentation allows us to enrich our dataset to prevent overfitting
        
        from keras.preprocessing.image import ImageDataGenerator
        
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                         target_size=(64, 64),
                                                         batch_size=32,
                                                         class_mode='binary')
        
        test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')
        
        classifier.fit_generator(training_set,
                                samples_per_epoch=8000,
                                nb_epoch=25,
                                validation_data=test_set,
                                nb_val_samples=2000)
        
        # Part 3 - Making new predictions
        
        import numpy as np
        from keras.preprocessing import image
        test_image = image.load_img('dataset/single_prediction/', target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        training_set.class_indices
        if result[0][0] == 1: 
            prediction = 'suspicious'
            activity_name="."
        else:
            prediction = 'Non-suspicious'
            activity_name="."
    return 0  
    activity_name=""      
# 3'rd gaussian is most probable and 1'st gaussian is least probable
def class_name(activity_class):
    if activity_class==1:
        activity_name="Walking Detection!!!"
        predict = "Walking Detection!!!"
    elif activity_class==2:
        activity_name="Jogging Detection!!!"
        predict = "Walking Detection!!!"
    elif activity_class==3:
        activity_name="Clapping Detection!!!"
        predict = "Walking Detection!!!"
    elif activity_class==4:
        activity_name="Fighting Detection--> suspicious"
        predict = "Walking Detection!!!"
    elif activity_class==5:
        activity_name="Normal Crowd Detection!!"
        predict = "Walking Detection!!!"
    elif activity_class==6:
        activity_name="Abnormal Crowd Detection-->suspicious"
        predict = "Walking Detection!!!"
    return activity_name

# creating object 
fgbg1 = cv2.createBackgroundSubtractorMOG2();    
fgbg2 = cv2.createBackgroundSubtractorMOG2(); 
fgbg3 = cv2.createBackgroundSubtractorMOG2(); 
cap = cv2.VideoCapture('chain.mp4')
_,frame1 = cap.read()
frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

# getting shape of the frame
row,col = frame.shape

# initialising mean,variance,omega and omega by sigma
mean = np.zeros([3,row,col],np.float64)
mean[1,:,:] = frame

variance = np.zeros([3,row,col],np.float64)
variance[:,:,:] = 400

omega = np.zeros([3,row,col],np.float64)
omega[0,:,:],omega[1,:,:],omega[2,:,:] = 0,0,1

omega_by_sigma = np.zeros([3,row,col],np.float64)

# initialising foreground and background
foreground = np.zeros([row,col],np.uint8)
background = np.zeros([row,col],np.uint8)

#initialising T and alpha
alpha = 0.3
T = 0.5

# converting data type of integers 0 and 255 to uint8 type
a = np.uint8([255])
b = np.uint8([0])
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
learning_rate=0
learning_rate,start_point1,start_point,end_point,predict=connectedLayers(alpha,length,frame1)

# Check if video opened successfully 
if (cap.isOpened()== False):  
  print("Error opening video  file") 
count=0   
c=0
while(cap.isOpened()): 
    ret, frame2 = cap.read() 
    if learning_rate==0: 
        frame_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
        # converting data type of frame_gray so that different operation with it can be performed
        frame_gray = frame_gray.astype(np.float64)
    
        # Because variance becomes negative after some time because of norm_pdf function so we are converting those indices 
        # values which are near zero to some higher values according to their preferences
        variance[0][np.where(variance[0]<1)] = 10
        variance[1][np.where(variance[1]<1)] = 5
        variance[2][np.where(variance[2]<1)] = 1
    
        #calulating standard deviation
        sigma1 = np.sqrt(variance[0])
        sigma2 = np.sqrt(variance[1])
        sigma3 = np.sqrt(variance[2])
    
        # getting values for the inequality test to get indexes of fitting indexes
        compare_val_1 = cv2.absdiff(frame_gray,mean[0])
        compare_val_2 = cv2.absdiff(frame_gray,mean[1])
        compare_val_3 = cv2.absdiff(frame_gray,mean[2])
    
        value1 = 2.5 * sigma1
        value2 = 2.5 * sigma2
        value3 = 2.5 * sigma3
    
        # finding those indexes where values of T are less than most probable gaussian and those where sum of most probale
        # and medium probable is greater than T and most probable is less than T
        fore_index1 = np.where(omega[2]>T)
        fore_index2 = np.where(((omega[2]+omega[1])>T) & (omega[2]<T))
    
        # Finding those indices where a particular pixel values fits at least one of the gaussian
        gauss_fit_index1 = np.where(compare_val_1 <= value1)
        gauss_not_fit_index1 = np.where(compare_val_1 > value1)
    
        gauss_fit_index2 = np.where(compare_val_2 <= value2)
        gauss_not_fit_index2 = np.where(compare_val_2 > value2)
    
        gauss_fit_index3 = np.where(compare_val_3 <= value3)
        gauss_not_fit_index3 = np.where(compare_val_3 > value3)
    
        #finding common indices for those indices which satisfy line 70 and 80
        temp = np.zeros([row, col])
        temp[fore_index1] = 1
        temp[gauss_fit_index3] = temp[gauss_fit_index3] + 1
        index3 = np.where(temp == 2)
    
        # finding com
        temp = np.zeros([row,col])
        temp[fore_index2] = 1
        index = np.where((compare_val_3<=value3)|(compare_val_2<=value2))
        temp[index] = temp[index]+1
        index2 = np.where(temp==2)
    
        match_index = np.zeros([row,col])
        match_index[gauss_fit_index1] = 1
        match_index[gauss_fit_index2] = 1
        match_index[gauss_fit_index3] = 1
        not_match_index = np.where(match_index == 0)
    
        #updating variance and mean value of the matched indices of all three gaussians
        rho = alpha * norm_pdf(frame_gray[gauss_fit_index1], mean[0][gauss_fit_index1], sigma1[gauss_fit_index1])
        constant = rho * ((frame_gray[gauss_fit_index1] - mean[0][gauss_fit_index1]) ** 2)
        mean[0][gauss_fit_index1] = (1 - rho) * mean[0][gauss_fit_index1] + rho * frame_gray[gauss_fit_index1]
        variance[0][gauss_fit_index1] = (1 - rho) * variance[0][gauss_fit_index1] + constant
        omega[0][gauss_fit_index1] = (1 - alpha) * omega[0][gauss_fit_index1] + alpha
        omega[0][gauss_not_fit_index1] = (1 - alpha) * omega[0][gauss_not_fit_index1]
    
        rho = alpha * norm_pdf(frame_gray[gauss_fit_index2], mean[1][gauss_fit_index2], sigma2[gauss_fit_index2])
        constant = rho * ((frame_gray[gauss_fit_index2] - mean[1][gauss_fit_index2]) ** 2)
        mean[1][gauss_fit_index2] = (1 - rho) * mean[1][gauss_fit_index2] + rho * frame_gray[gauss_fit_index2]
        variance[1][gauss_fit_index2] = (1 - rho) * variance[1][gauss_fit_index2] + rho * constant
        omega[1][gauss_fit_index2] = (1 - alpha) * omega[1][gauss_fit_index2] + alpha
        omega[1][gauss_not_fit_index2] = (1 - alpha) * omega[1][gauss_not_fit_index2]
    
        rho = alpha * norm_pdf(frame_gray[gauss_fit_index3], mean[2][gauss_fit_index3], sigma3[gauss_fit_index3])
        constant = rho * ((frame_gray[gauss_fit_index3] - mean[2][gauss_fit_index3]) ** 2)
        mean[2][gauss_fit_index3] = (1 - rho) * mean[2][gauss_fit_index3] + rho * frame_gray[gauss_fit_index3]
        variance[2][gauss_fit_index3] = (1 - rho) * variance[2][gauss_fit_index3] + constant
        omega[2][gauss_fit_index3] = (1 - alpha) * omega[2][gauss_fit_index3] + alpha
        omega[2][gauss_not_fit_index3] = (1 - alpha) * omega[2][gauss_not_fit_index3]
        
        # updating least probable gaussian for those pixel values which do not match any of the gaussian
        mean[0][not_match_index] = frame_gray[not_match_index]
        variance[0][not_match_index] = 200
        omega[0][not_match_index] = 0.1
    
        # normalise omega
        sum = np.sum(omega,axis=0)
        omega = omega/sum
    
        #finding omega by sigma for ordering of the gaussian
        omega_by_sigma[0] = omega[0] / sigma1
        omega_by_sigma[1] = omega[1] / sigma2
        omega_by_sigma[2] = omega[2] / sigma3
    
        # getting index order for sorting omega by sigma
        index = np.argsort(omega_by_sigma,axis=0)
        
        # from that index(line 139) sorting mean,variance and omega
        mean = np.take_along_axis(mean,index,axis=0)
        variance = np.take_along_axis(variance,index,axis=0)
        omega = np.take_along_axis(omega,index,axis=0)
        activity_class=detect_activities(frame2,length) 
        activity_name=class_name(activity_class)
       
        # converting data type of frame_gray so that we can use it to perform operations for displaying the image
        frame_gray = frame_gray.astype(np.uint8)
        (regions, _) = hog.detectMultiScale(frame2, 
                                                winStride=(4, 4), 
                                                padding=(4, 4), 
                                                scale=1.05) 
       
            # Drawing the regions in the  
            # Image 
        for (x, y, w, h) in regions: 
                cv2.rectangle(frame2, (x, y), 
                              (x + w, y + h),  
                              (0, 0, 255), 2) 
                cv2.putText(frame2,activity_name,  (x, y+25) , cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 1)

            # Showing the output Image 
        print("Predicted Activity:")
        print(activity_name)
        current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        str_current_datetime = str(current_datetime)
        file = open("time.txt", 'a')
        #file.writelines(predict)
        file.writelines(activity_name)
        file.writelines(str_current_datetime + '\n')
        # getting background from the index2 and index3
        background[index2] = frame_gray[index2]
        background[index3] = frame_gray[index3]
        cv2.imshow('Background Subtraction',cv2.subtract(frame_gray,background))
        
        showInMovedWindow('Activity Detection_',frame2, 0, 200)
        time.sleep(0.1)
        count += 1
        if cv2.waitKey(1) & 0xFF == 27:
           break
    else:
          # apply mask for background subtraction 
        fgmask1 = fgbg1.apply(frame2); 
        fgmask2 = fgbg2.apply(frame2); 
        fgmask3 = fgbg3.apply(frame2); 
         
        
    # vertically concatenates images  
    # of same width 
       
         # Black color in BGR 
        color = (255, 0, 0) 
        cv2.imshow('OUTPUT', fgmask2) 
        
    # Line thickness of -1 px 
    # Thickness of -1 will fill the entire shape 
        thickness = 4
        learning_rate,start_point1,start_point,end_point,predict=connectedLayers(c,length,fgmask3)
        #print(predict)
        image = cv2.rectangle(frame1, start_point, end_point, color, thickness)
        cv2.putText(image, predict, (start_point1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow('Activity Detection_',frame2)
        c=c+1
        print("Predicted Activity:")
        print(predict)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    str_current_datetime = str(current_datetime)
    file = open("time.txt", 'a')
    file.writelines(predict)
    #file.writelines(activity_name)
    file.writelines(str_current_datetime + '\n')
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break; 
  

