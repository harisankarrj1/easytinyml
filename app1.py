# app1.py

def set_labels(new_labels):
    with open('labels.txt', 'w') as file:
        file.write(','.join(new_labels))

def get_labels():
    try:
        with open('labels.txt', 'r') as file:
            labels = file.read().split(',')
            return [label.strip() for label in labels if label.strip()]
    except FileNotFoundError:
        return []

# Rest of your app1.py code...
