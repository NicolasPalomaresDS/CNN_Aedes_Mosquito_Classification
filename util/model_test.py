import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def get_misclassification_examples(y_test, predictions, test_images, class_names):
    actuals = []
    misclassified_images = [i for i, x in enumerate(y_test != predictions) if x==True]
    plt.figure(figsize=(10,10))

    for i, number in enumerate(misclassified_images[:9]):
        ax = plt.subplot(3, 3, i+1)
        img = test_images[number]
        # img = (img * -255).astype('uint8')
        plt.imshow(img.astype('uint8'))
        plt.axis("off")
        
        actual_label = class_names[int(y_test[number])]
        pred_label = class_names[int(predictions[number])]
        actuals.append(actual_label)
        
        ax.set_title(
            f"Real: {actual_label}\nPred: {pred_label}", 
            fontsize=10, 
            color="red", 
            loc="center",
            y=-0.20
        )
    plt.show()
    
    

def get_results(model, dataset):
    all_images = []
    y_test = np.array([])
    pred_test = None

    for x, y in dataset:
        y_test = np.concatenate([y_test, y])
        preds = model.predict(x, verbose=0)
        if pred_test is None:
            pred_test = preds
        else:
            pred_test = np.concatenate([pred_test, preds], axis=0)
        all_images.append(x)

    test_images = np.concatenate(all_images)
    predictions = np.argmax(pred_test, axis=1)
    class_names = dataset.class_names
    y_test_labels = [class_names[int(i)] for i in y_test]
    pred_labels = [class_names[int(i)] for i in predictions]

    print("Classification Report:\n")
    print(classification_report(y_test_labels, pred_labels, target_names=class_names))

    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True)
    plt.title('Confussion Matrix')
    plt.show()
    
    get_misclassification_examples(y_test, predictions, test_images, class_names)
    