import Preprocess as pp





if __name__ == "__mian__":
    test = '/Users/ericwu/python/af/sample2017/validation'
    train = '/Users/ericwu/python/af/training2017'
    _, _, _, _ = pp.PrepareFeatures(train, test)
    pp.get_MedAmpInput() #okay
    pp.get_TempInput() #okay
    pp.get_SpecgInput() #okay
 