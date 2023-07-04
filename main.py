import gender



#model 1 - 20 epochs batch size 32 lr 0.001
# classifier = gender.GenderCNN(epochs=20, batch_size=32, lr=0.001)
# classifier.load_images()
# classifier.build_model1()
# classifier.train_model()
# classifier.evaluate_model()

#model 2 - 20 epochs batch size 32 lr 0.001
# classifier = gender.GenderCNN(epochs=20, batch_size=32, lr=0.001)
# classifier.load_images()
# classifier.build_model2()
# classifier.train_model()
# classifier.evaluate_model()

#model 3 - 20 epochs batch size 32 lr 0.001
# classifier = gender.GenderCNN(epochs=20, batch_size=32, lr=0.001)
# classifier.load_images()
# classifier.build_model3()
# classifier.train_model()
# classifier.evaluate_model()


#model 1 - 20 epochs batch size 64 lr 0.001
# classifier = gender.GenderCNN(epochs=20, batch_size=64, lr=0.001)
# classifier.load_images()
# classifier.build_model1()
# classifier.train_model()
# classifier.evaluate_model()

#model 2 - 20 epochs batch size 64 lr 0.001
# classifier = gender.GenderCNN(epochs=20, batch_size=32, lr=0.001)
# classifier.load_images()
# classifier.build_model2()
# classifier.train_model()
# classifier.evaluate_model()

#model 3 - 20 epochs batch size 64 lr 0.001
classifier = gender.GenderCNN(epochs=20, batch_size=64, lr=0.001)
classifier.load_images()
classifier.build_model3()
classifier.train_model()
classifier.evaluate_model()

# classifier.train_model()
# classifier.evaluate_model()
# classifier.save_model()
# classifier.load_model(2)
classifier.save_model()
gender = classifier.predict_gender('/Users/nikolasehovac/Desktop/IMG_5203.PNG')

print(gender)
