import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
import cv2
import mahotas as mt
from flask import *

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/prediction/"

def training_phase():
    ### Reading the dataset
    dataset = pd.read_csv("models/Flavia_features.csv")
    dataset.head(5)
    type(dataset)
    ds_path = "static/Leaves"
    img_files = os.listdir(ds_path)

    ### Creating target labels
    breakpoints = [1001,1059,1060,1122,1552,1616,1123,1194,1195,1267,1268,1323,1324,1385,1386,1437,1497,1551,1438,1496,2001,2050,2051,2113,2114,2165,2166,2230,2231,2290,2291,2346,2347,2423,2424,2485,2486,2546,2547,2612,2616,2675,3001,3055,3056,3110,3111,3175,3176,3229,3230,3281,3282,3334,3335,3389,3390,3446,3447,3510,3511,3563,3566,3621]
    target_list = []
    for file in img_files:
        target_num = int(file.split(".")[0])
        flag = 0
        i = 0 
        for i in range(0,len(breakpoints),2):
            if((target_num >= breakpoints[i]) and (target_num <= breakpoints[i+1])):
                flag = 1
                break
        if(flag==1):
            target = int((i/2))
            target_list.append(target)
    y = np.array(target_list)
    y
    X = dataset.iloc[:,1:]
    X.head(5)
    y[0:5]

    ### Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 142)
    X_train.head(5)
    y_train[0:5]

    ### Feature Scaling
    training_phase.sc_X = StandardScaler()
    X_train = training_phase.sc_X.fit_transform(X_train)
    X_test = training_phase.sc_X.transform(X_test)
    X_train[0:2]
    y_train[0:2]

    ### Applying SVM classifier model
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    metrics.accuracy_score(y_test, y_pred)
    # print(metrics.classification_report(y_test, y_pred))

    ### Performing parameter tuning of the model
    parameters = [{'kernel': ['rbf'],
                'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                'C': [1, 10, 100, 1000]},
                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
                ]
    training_phase.svm_clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
    training_phase.svm_clf.fit(X_train, y_train)
    training_phase.svm_clf.best_params_
    means = training_phase.svm_clf.cv_results_['mean_test_score']
    stds = training_phase.svm_clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, training_phase.svm_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    y_pred_svm = training_phase.svm_clf.predict(X_test)
    metrics.accuracy_score(y_test, y_pred_svm)
    # print(metrics.classification_report(y_test, y_pred_svm))

    ### Dimensionality Reduction using PCA
    pca = PCA()
    pca.fit(X)
    var= pca.explained_variance_ratio_
    var
    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    # plt.plot(var1)

training_phase()

@app.route('/')
def display():
    # training_phase()
    return render_template("index.html")

@app.route('/result', methods = ['POST'])
def result():
    f = request.files['file']
    f.save(app.config['UPLOAD_FOLDER'] + f.filename)
    new_path = "static/prediction/" + f.filename
    return render_template("result.html", clasi = classify(f.filename), img = new_path, p_img = classify.plant_img)
   
def classify(f_name):
    # Testing wala main part
    def bg_sub(filename):
        test_img_path = 'static/prediction/' + filename
        main_img = cv2.imread(test_img_path)
        img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img, (1600, 1200))
        size_y,size_x,_ = img.shape
        gs = cv2.cvtColor(resized_image,cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gs, (55,55),0)
        ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((50,50),np.uint8)
        closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
        
        contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contains = []
        y_ri,x_ri, _ = resized_image.shape
        for cc in contours:
            yn = cv2.pointPolygonTest(cc,(x_ri//2,y_ri//2),False)
            contains.append(yn)

        val = [contains.index(temp) for temp in contains if temp>0]
        index = val[0]
        
        black_img = np.empty([1200,1600,3],dtype=np.uint8)
        black_img.fill(0)
        
        cnt = contours[index]
        mask = cv2.drawContours(black_img, [cnt] , 0, (255,255,255), -1)
        
        maskedImg = cv2.bitwise_and(resized_image, mask)
        white_pix = [255,255,255]
        black_pix = [0,0,0]
        
        final_img = maskedImg
        h,w,channels = final_img.shape
        for x in range(0,w):
            for y in range(0,h):
                channels_xy = final_img[y,x]
                if all(channels_xy == black_pix):
                    final_img[y,x] = white_pix
        
        return final_img

    filename = f_name
    bg_rem_img = bg_sub(filename)

    # plt.imshow(bg_rem_img)

    def feature_extract(img):
        names = ['area','perimeter','pysiological_length','pysiological_width','aspect_ratio','rectangularity','circularity', \
                'mean_r','mean_g','mean_b','stddev_r','stddev_g','stddev_b', \
                'contrast','correlation','inverse_difference_moments','entropy'
                ]
        df = pd.DataFrame([], columns=names)

        #Preprocessing
        gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gs, (25,25),0)
        ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((50,50),np.uint8)
        closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

        #Shape features
        contours, hierarchy= cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        M = cv2.moments(cnt, True)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        rectangularity = w*h/area
        circularity = ((perimeter)**2)/area

        #Color features
        red_channel = img[:,:,0]
        green_channel = img[:,:,1]
        blue_channel = img[:,:,2]
        blue_channel[blue_channel == 255] = 0
        green_channel[green_channel == 255] = 0
        red_channel[red_channel == 255] = 0

        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)

        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)

        #Texture features
        textures = mt.features.haralick(gs)
        ht_mean = textures.mean(axis=0)
        contrast = ht_mean[1]
        correlation = ht_mean[2]
        inverse_diff_moments = ht_mean[4]
        entropy = ht_mean[8]

        vector = [area,perimeter,w,h,aspect_ratio,rectangularity,circularity,\
                red_mean,green_mean,blue_mean,red_std,green_std,blue_std,\
                contrast,correlation,inverse_diff_moments,entropy
                ]

        df_temp = pd.DataFrame([vector],columns=names)
        df = df.append(df_temp)
        
        return df
        
    features_of_img = feature_extract(bg_rem_img)
    features_of_img

    scaled_features = training_phase.sc_X.transform(features_of_img)
    # print(scaled_features)
    y_pred_mobile = training_phase.svm_clf.predict(scaled_features)
    y_pred_mobile[0]

    common_names = ['Pubescent Bamboo','Chinese Horse Chestnut','Anhui Barberry', \
                    'Chinese Redbud','True Indigo','Japanese Maple','Nanmu','Castor Aralia', \
                    'Chinese Cinnamon','Goldenrain Tree','Big-fruited Holly','Japanese Cheesewood', \
                    'Wintersweet','Camphortree','Japan Arrowwood','Sweet Osmanthus','Deodar','Ginkgo, Maidenhair Tree', \
                    'Crape Myrtle, Crepe Myrtle','Oleander','Yew Plum Pine','Japanese Flowering Cherry','Glossy Privet',\
                    'Chinese Toon','Peach','Ford Woodlotus','Trident Maple','Beales Barberry','Southern Magnolia',\
                    'Canadian Poplar','Chinese Tulip Tree','Tangerine'
                ]
    f_type = common_names[y_pred_mobile[0]]

    plant_dict = {
        'Pubescent Bamboo': 'static/plants/pubescent bamboo.jpg',
        'Chinese Horse Chestnut': 'static/plants/Chinese Horse Chestnut.jpg',
        'Anhui Barberry': 'static/plants/Anhui Barberry.jpg',
        'Chinese Redbud': 'static/plants/Chinese Redbud.jpg',
        'True Indigo': 'static/plants/True Indigo.jpg',
        'Japanese Maple': 'static/plants/Japanese Maple.jpg',
        'Nanmu': 'static/plants/Nanmu.jpeg',
        'Castor Aralia': 'static/plants/Castor Aralia.jpg',
        'Chinese Cinnamon': 'static/plants/Chinese Cinnamon.jpg',
        'Goldenrain Tree': 'static/plants/Goldenrain Tree.jpg',
        'Big-fruited Holly': 'static/plants/Big-fruited Holly.jpg',
        'Japanese Cheesewood': 'static/plants/Japanese Cheesewood.jpg',
        'Wintersweet': 'static/plants/Winter Sweet.jpg',
        'Camphortree': 'static/plants/Camphortree.jpeg',
        'Japan Arrowwood': 'static/plants/Japanese Arrowwood.jpeg',
        'Sweet Osmanthus': 'static/plants/Sweet Osmanthus.jpg',
        'Deodar': 'static/plants/deodar.jpg',
        'Ginkgo, Maidenhair Tree': 'static/plants/Maidenhair Tree.jpeg',
        'Crape Myrtle, Crepe Myrtle': 'static/plants/Crape Myrtle, Crepe Myrtle.JPG',
        'Oleander': 'static/plants/Oleander.jpg',
        'Yew Plum Pine': 'static/plants/Yew Plum Pine.jpg',
        'Japanese Flowering Cherry': 'static/plants/Japanese Flowering Cherry.jpg',
        'Glossy Privet': 'static/plants/Glossy Privet.jpg',
        'Chinese Toon': 'static/plants/Chinese Toon.jpg',
        'Peach': 'static/plants/Peach.jpg',
        'Ford Woodlotus': 'static/plants/Ford Woodlotus Plant.jpg',
        'Trident Maple': 'static/plants/Trident Maple.jpg',
        'Beales Barberry': 'static/plants/Beales Barberry.jpg',
        'Southern Magnolia': 'static/plants/Southern Magnolia.jpg',
        'Canadian Poplar': 'static/plants/Canadian Poplar.jpg',
        'Chinese Tulip Tree': 'static/plants/Chinese Tulip Tree.jpg',
        'Tangerine': 'static/plants/Tangerine.jpg'
        }

    classify.plant_img = plant_dict[f_type]

    return f_type
    
if __name__ == "__main__":
    app.run(debug = True)