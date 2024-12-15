import cv2

from extract_symbol.extract_symbol import extract_symbol
from recognition.predict import load_model, predict

# Read the image from the specified path
image_path = "./data/package_aa/000100134-10_1_1.png"  # Replace with your image path
img = cv2.imread(image_path)

# Split the image into parts
split_symbols = extract_symbol(img)

def get_actl_prediction(prediction):
    identifiers = ['4', '8', '16', '32', '#', 'bar', 'clef', 'flat', 'natural', 'chord', 'dot', 'p']
    for i in identifiers:
        if i in prediction:
            return i

def get_note_types(split_symbols):
    predict_result = []
    # Predict for each split image
    for i, row in enumerate(split_symbols):
        predict_result_row = []
        for j, symbol in enumerate(row):
            cv2.imwrite(f"./outputs/split_notes/processed_{i}_{j}.png", symbol)
            model = load_model("./model/NN_model.pkl")
            prediction = predict(symbol, model)
            prediction = get_actl_prediction(prediction)
            predict_result_row.append(prediction)
        predict_result.append(predict_result_row)
    return predict_result

predict_result = get_note_types(split_symbols)
    

print("Predictions:")
for i, row in enumerate(predict_result):
    print(f"Row {i}: {row}")
