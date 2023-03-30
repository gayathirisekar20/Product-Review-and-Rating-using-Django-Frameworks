from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import *
from .forms import *
from django.db.models import Avg
from distutils.log import debug
#from urllib import request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('omw-1.4')
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
wordnet_lemmatizer = WordNetLemmatizer()
# import train_model 
# Create your views here.
def home(request):
    query = request.GET.get("title")
    allproducts = None
    if query:
        allproducts = Product.objects.filter(name__icontains=query)
    else:	
        allproducts = Product.objects.all()
    context = {
        "product":allproducts,
    }
    return render(request,'main/index.html',context)

def detail(request,id):
    product=Product.objects.get(id=id)
    reviews = Review.objects.filter(product=id).order_by("-comment")
    # print(reviews)
    pos=0
    neg=0
    pos_msg=[]
    neg_msg=[]
    msg=[]
    for review in reviews :
                print(review.comment)
                # print(type(review.comment))
                BASE_DI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                MODEL_ROOT = os.path.join(BASE_DI, "models")
                model_name='amazon_data.txt'
                # train_name='train_model.py'
                clf=os.path.join(MODEL_ROOT, model_name)
                df = pd.read_csv(clf, names=['review', 'sentiment'], sep='\t') 
                
                def normalizer(tweet):
                    only_letters = re.sub("[^a-zA-Z]", " ", tweet)
                    only_letters = only_letters.lower()
                    only_letters = only_letters.split()
                    filtered_result = [word for word in only_letters if word not in stopwords.words('english')]
                    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
                    lemmas = ' '.join(lemmas)
                    return lemmas
                df = shuffle(df)
                y = df['sentiment']
                x = df.review.apply(normalizer)

                vectorizer = CountVectorizer()
                x_vectorized = vectorizer.fit_transform(x)

                train_x,val_x,train_y,val_y = train_test_split(x_vectorized,y)

                regressor = LogisticRegression(multi_class='multinomial', solver='newton-cg')
                # print('process finish')
                model = regressor.fit(train_x, train_y)

                params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
                gs_clf = GridSearchCV(model, params, n_jobs=1, cv=5)
                gs_clf = gs_clf.fit(train_x, train_y)
                model = gs_clf.best_estimator_

                y_pred = model.predict(val_x)

                _f1 = f1_score(val_y, y_pred, average='micro')
                _confusion = confusion_matrix(val_y, y_pred)
                __precision = precision_score(val_y, y_pred, average='micro')
                _recall = recall_score(val_y, y_pred, average='micro')
                _statistics = {'f1_score': _f1,
                                'confusion_matrix': _confusion,
                                'precision': __precision,
                                'recall': _recall
                                }
                # print (_statistics)
                rev= review.comment
                # print(type(rev),rev)
                test_feature = vectorizer.transform([rev])
                out=model.predict(test_feature)
                out1=out[0]
                # print(out1)
                
                if out1==1:
                    pos=pos+1
                    pos_msg.append(review.comment)
                    msg.append("positive")
                else :
                    neg_msg.append(review.comment)	
                    neg= neg+1
                    msg.append("negative")
                 
    rev=len(reviews)                
    print('pos:',pos)	
    print('neg:',neg)
    print('msg',msg)
    if rev:
        rate= round((pos/rev)*5,2)
        print(rate)  
    else:
        rate=0      
    print(pos_msg)
    print(neg_msg)
    print(len(reviews))
    
    average = reviews.aggregate(Avg("rating"))["rating__avg"]
    if average == None:
        average=0
    else:
        average = round(average,2)
    context={
        "prod":product,
        "reviews":reviews,
        "average":average,
        "rate":rate,
        "pos_msg":pos_msg,
        "neg_msg":neg_msg,
        "msg":msg,
    }
    return render(request,'main/details.html',context)

def add_products(request):
    if request.user.is_authenticated:
        if request.user.is_superuser:
            if request.method == "POST":
                form = ProductForm(request.POST or None)

                if form.is_valid():
                    data = form.save(commit=False)
                    data.save()
                    return redirect("main:home")
            else:
                form = ProductForm()
            return render(request,'main/addproducts.html',{"form":form,"controller":"Add Products"})
    	# else:
    	# 	return redirect("main:home")
    return redirect("accounts:login")

def edit_products(request,id):
    if request.user.is_authenticated:
        if request.user.is_superuser:
            produ = Product.objects.get(id=id)
            if request.method == "POST":
                form = ProductForm(request.POST or None, instance=produ)
                if form.is_valid():
                    data = form.save(commit=False)
                    data.save()
                    return redirect("main:detail",id)
            else:
                form =ProductForm(instance=produ)
                return render(request,'main/addproducts.html',{'form':form, "controller":"Edit Products"})
        
    return redirect("accounts:login")

def delete_products(request,id):
    if request.user.is_authenticated:
        if request.user.is_superuser:
            produ = Product.objects.get(id=id)
            produ.delete()
            return redirect("main:home")
        else:
            return redirect("accounts:login")

def add_review(request,id) :
    if request.user.is_authenticated:
        product = Product.objects.get(id=id)
        if request.method == "POST":
            form = ReviewForm(request.POST or None)
            if form.is_valid():
                data = form.save(commit=False)
                data.comment = request.POST["comment"]
                # print(len(data.comment))
                print(data.comment)
                BASE_DI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                MODEL_ROOT = os.path.join(BASE_DI, "models")
                model_name='amazon_data.txt'
                # train_name='train_model.py'
                clf=os.path.join(MODEL_ROOT, model_name)
                df = pd.read_csv(clf, names=['review', 'sentiment'], sep='\t') 
                
                def normalizer(tweet):
                    only_letters = re.sub("[^a-zA-Z]", " ", tweet)
                    only_letters = only_letters.lower()
                    only_letters = only_letters.split()
                    filtered_result = [word for word in only_letters if word not in stopwords.words('english')]
                    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
                    lemmas = ' '.join(lemmas)
                    return lemmas
                df = shuffle(df)
                y = df['sentiment']
                x = df.review.apply(normalizer)

                vectorizer = CountVectorizer()
                x_vectorized = vectorizer.fit_transform(x)

                train_x,val_x,train_y,val_y = train_test_split(x_vectorized,y)

                regressor = LogisticRegression(multi_class='multinomial', solver='newton-cg')
                print('process finish')
                model = regressor.fit(train_x, train_y)

                params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
                gs_clf = GridSearchCV(model, params, n_jobs=1, cv=5)
                gs_clf = gs_clf.fit(train_x, train_y)
                model = gs_clf.best_estimator_

                y_pred = model.predict(val_x)

                _f1 = f1_score(val_y, y_pred, average='micro')
                _confusion = confusion_matrix(val_y, y_pred)
                __precision = precision_score(val_y, y_pred, average='micro')
                _recall = recall_score(val_y, y_pred, average='micro')
                _statistics = {'f1_score': _f1,
                                'confusion_matrix': _confusion,
                                'precision': __precision,
                                'recall': _recall
                                }
                # print (_statistics)
                rev= data.comment
                print(type(rev),rev)
                test_feature = vectorizer.transform([rev])
                out=model.predict(test_feature)
                out1=out[0]
                print(out1)
                
                if out1==1:
                    print('pos')
                else:
                    print('neg')
                # data.rating = request.POST["rating"]
                data.rating=0
                data.user = request.user
                data.product = product
                data.save()
                return redirect("main:detail",id)
        else:
            form = ReviewForm()
        return render(request,'main/details.html',{'form':form})
    else:
        return redirect("accounts:login")

def edit_review(request,product_id,review_id):
    if request.user.is_authenticated:
        product = Product.objects.get(id=product_id)
        review = Review.objects.get(product=product, id=review_id)
        if request.user == review.user:
            if request.method == "POST":
                form = ReviewForm(request.POST,instance=review)
                if form.is_valid():
                    data = form.save(commit=False)
                    if (data.rating>10) or (data.rating<0):
                        error="Out of range. Please select rating from 0 to 10."
                        return render(request,'main/editreview.html',{'error':error, "form":form})
                    else:
                        data.save()
                        return redirect("main:detail",product_id)
            else:
                form = ReviewForm(instance=review)
            return render(request,'main/editreview.html',{"form":form})
        else:
            return redirect("main:detail",product_id)
    else:
        return redirect("accounts:login")

def delete_review(request,product_id,review_id):
    if request.user.is_authenticated:
        product = Product.objects.get(id=product_id)
        review = Review.objects.get(product=product, id=review_id)
        if request.user == review.user:
            review.delete()
        return redirect("main:detail",product_id)
    else:
        return redirect("accounts:login")

def about(request):
    return render(request,'main/about.html')