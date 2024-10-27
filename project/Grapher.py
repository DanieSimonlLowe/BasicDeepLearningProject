
def save_loss(file,epoch,train_loss,test_loss):
    with open(file, "a") as file:
    # Append a line to the file
        file.write(f"{epoch},{train_loss},{test_loss}\n")
        print(f"{epoch},{train_loss},{test_loss}\n")