from torchvision.io import ImageReadMode, read_image

if __name__ == "__main__":

    test_img = read_image("test.jpg", mode=ImageReadMode.RGB)
    print("\n\nTEST IMAGE : \n\n", test_img, "\n\nShape: ", test_img.shape)
