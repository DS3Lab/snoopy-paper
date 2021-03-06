empty = {
    "256": 0.0,
    "512": 0.0,
    "1024": 0.0,
    "2048": 0.0,
    "4096": 0.0,
    "8192": 0.0,
    "16384": 0.0,
}

# Means of 5 consecutive measurements in seconds
image_result = {
    "empty": empty,
    "raw": empty,
    "pca_32": empty,
    "pca_64": empty,
    "pca_128": empty,
    "alexnet": {
        "1024": 1.7725129777565598,
        "2048": 4.108953144587576,
        "512": 0.6176360009238124,
    },
    "efficientnet_b0": {
        "1024": 2.9459043156355618,
        "2048": 3.8309638269245623,
        "512": 1.2831428924575448,
    },
    "efficientnet_b1": {
        "1024": 3.666610764153302,
        "2048": 6.03861366789788,
        "512": 2.4720458345487715,
    },
    "efficientnet_b2": {
        "1024": 5.984911478683353,
        "2048": 8.95302885659039,
        "512": 4.397853007540107,
    },
    "efficientnet_b3": {
        "1024": 5.163835605233908,
        "2048": 9.925222208350897,
        "512": 2.776769632473588,
    },
    "efficientnet_b4": {
        "1024": 9.544669179804623,
        "2048": 18.69876848626882,
        "512": 5.003797982633114,
    },
    "efficientnet_b5": {
        "1024": 19.33195194527507,
        "2048": 34.89768644534051,
        "512": 10.800818342901767,
    },
    "efficientnet_b6": {
        "1024": 27.395731732994317,
        "2048": 54.48761784750968,
        "512": 13.84335319828242,
    },
    "efficientnet_b7": {
        "1024": 50.786453934572634,
        "2048": 101.07741778511554,
        "512": 25.79525070041418,
    },
    "googlenet": {
        "1024": 1.9009166369214654,
        "2048": 4.151035995781422,
        "512": 0.744047917611897,
    },
    "inception": {
        "1024": 5.245462962985039,
        "2048": 8.350767710618674,
        "512": 3.6580580255016684,
    },
    "resnet_101_v2": {
        "1024": 5.610135013051331,
        "2048": 8.407415508292615,
        "512": 4.156196189299226,
    },
    "resnet_152_v2": {
        "1024": 4.992349454388022,
        "2048": 8.744984836317599,
        "512": 3.1282933352515103,
    },
    "resnet_50_v2": {
        "1024": 4.871747286617756,
        "2048": 6.926061735861003,
        "512": 3.889030870422721,
    },
    "vgg16": {
        "1024": 5.564772844873369,
        "2048": 10.97800384312868,
        "512": 2.996711520291865,
    },
    "vgg19": {
        "1024": 6.853968546167016,
        "2048": 13.01373979691416,
        "512": 3.647686574794352,
    },
}

times = {
    "cifar10": image_result,
    "cifar100": image_result,
    "mnist": image_result,
    "imdb_reviews": {
        "empty": empty,
        "bert_cased": {
            "1024": 23.416232520528137,
            "256": 11.135857886075973,
            "512": 15.753340835683048,
        },
        "bert_cased_large": {
            "1024": 53.53209322988987,
            "256": 16.731546410173177,
            "512": 28.64088070075959,
        },
        "bert_cased_large_pool": {
            "1024": 52.2600826125592,
            "256": 16.487148913368582,
            "512": 28.480767828971146,
        },
        "bert_cased_pool": {
            "1024": 23.75859120991081,
            "256": 11.832589769177138,
            "512": 15.498805034905672,
        },
        "bert_uncased": {
            "1024": 23.7744743719697,
            "256": 12.198540941625833,
            "512": 16.071898953989148,
        },
        "bert_uncased_large": {
            "1024": 53.55449336767197,
            "256": 16.412800670601428,
            "512": 29.079485500790177,
        },
        "bert_uncased_large_pool": {
            "1024": 52.45915455128998,
            "256": 16.657232538610696,
            "512": 28.689003485627474,
        },
        "bert_uncased_pool": {
            "1024": 23.974009062349797,
            "256": 12.321623141691088,
            "512": 16.104741353355347,
        },
        "elmo": {
            "1024": 250.5606592575088,
            "256": 62.02517025712878,
            "512": 118.88611205406487,
        },
        "nnlm_128": {
            "1024": 0.3220349822193384,
            "256": 0.2371202867478132,
            "512": 0.2684931244701147,
        },
        "nnlm_128_normalization": {
            "1024": 0.324151823669672,
            "256": 0.265145450271666,
            "512": 0.2956031773239374,
        },
        "nnlm_50": {
            "1024": 0.28926757983863355,
            "256": 0.2527977008372545,
            "512": 0.2542789263650775,
        },
        "nnlm_50_normalization": {
            "1024": 0.3723783878609538,
            "256": 0.26550060641020534,
            "512": 0.26201314497739076,
        },
        "use": {
            "1024": 1.5895505409687758,
            "256": 0.570133663341403,
            "512": 0.8800927920266985,
        },
        "use_large": {
            "1024": 7.775212369300425,
            "256": 2.002843094430864,
            "512": 3.876918362826109,
        },
        "xlnet": {
            "1024": 82.98753757346421,
            "256": 23.7838557748124,
            "512": 43.56562861762941,
        },
        "xlnet_large": {
            "1024": 214.67242330238224,
            "256": 56.22986972294748,
            "512": 109.1639247449115,
        },
    },
    "sst2": {
        "empty": empty,
        "bert_cased": {
            "1024": 17.50538479425013,
            "2048": 33.61235289573669,
            "512": 9.343297322839499,
        },
        "bert_cased_large": {
            "1024": 49.58935023564845,
            "2048": 97.91327755637467,
            "512": 25.460240580141544,
        },
        "bert_cased_large_pool": {
            "1024": 49.28309540860355,
            "2048": 97.03383858148008,
            "512": 25.623602638952434,
        },
        "bert_cased_pool": {
            "1024": 17.097552744671702,
            "2048": 33.21244122739881,
            "512": 9.146325661800802,
        },
        "bert_uncased": {
            "1024": 17.31969309784472,
            "2048": 34.428761855885384,
            "512": 9.114558858796954,
        },
        "bert_uncased_large": {
            "1024": 49.68697541262954,
            "2048": 98.01352554280311,
            "512": 25.434121181443334,
        },
        "bert_uncased_large_pool": {
            "1024": 49.38968270160258,
            "2048": 97.2117674453184,
            "512": 25.60155225340277,
        },
        "bert_uncased_pool": {
            "1024": 17.359144590608775,
            "2048": 32.93631013911217,
            "512": 9.130315243825317,
        },
        "elmo": {
            "1024": 3.352232603356242,
            "2048": 6.1825278045609595,
            "512": 1.9686481392011046,
        },
        "nnlm_128": {
            "1024": 0.4550862146541476,
            "2048": 0.4631916815415025,
            "512": 0.428352534212172,
        },
        "nnlm_128_normalization": {
            "1024": 0.4233076088130474,
            "2048": 0.4452163984999061,
            "512": 0.41775818113237617,
        },
        "nnlm_50": {
            "1024": 0.42356059458106754,
            "2048": 0.4298675190657377,
            "512": 0.4333739759400487,
        },
        "nnlm_50_normalization": {
            "1024": 0.429529413767159,
            "2048": 0.44865548703819513,
            "512": 0.4197696536779404,
        },
        "use": {
            "1024": 0.4510378960520029,
            "2048": 0.5490833861753345,
            "512": 0.4121444979682565,
        },
        "use_large": {
            "1024": 0.5131425153464079,
            "2048": 0.7255973137915135,
            "512": 0.45817717239260675,
        },
        "xlnet": {
            "1024": 77.86900285035372,
            "2048": 154.88061813004316,
            "512": 39.18239407353103,
        },
        "xlnet_large": {
            "1024": 208.32463576085865,
            "2048": 416.9482963791117,
            "512": 104.62408320754767,
        },
    },
    "yelp": {
        "empty": empty,
        "bert_cased": {
            "16384": 275.0499818595126,
            "4096": 71.24043523333967,
            "8192": 139.25740625206382,
        },
        "bert_cased_large": {
            "16384": 777.8163683958352,
            "4096": 198.3834778783843,
            "8192": 391.10755375046284,
        },
        "bert_cased_large_pool": {
            "16384": 763.1054892523214,
            "4096": 193.39015848301352,
            "8192": 383.0065444631502,
        },
        "bert_cased_pool": {
            "16384": 263.851340335235,
            "4096": 68.61551191341132,
            "8192": 133.4275134040043,
        },
        "bert_uncased": {
            "16384": 273.670351405628,
            "4096": 70.83487220015377,
            "8192": 137.09433097746222,
        },
        "bert_uncased_large": {
            "16384": 780.1785911055282,
            "4096": 196.97177666984499,
            "8192": 391.28468163758515,
        },
        "bert_uncased_large_pool": {
            "16384": 768.2800533538684,
            "4096": 193.57736848499627,
            "8192": 383.5071927251294,
        },
        "bert_uncased_pool": {
            "16384": 264.16920342333617,
            "4096": 69.53864413164555,
            "8192": 134.45364474374801,
        },
        "elmo": {
            "16384": 1401.278177678585,
            "4096": 351.3379330910742,
            "8192": 706.9672367095948,
        },
        "nnlm_128": {
            "16384": 1.5637276504188775,
            "4096": 0.9978711692616343,
            "8192": 1.1236314000561833,
        },
        "nnlm_128_normalization": {
            "16384": 1.6525195943191648,
            "4096": 0.9656130857765675,
            "8192": 1.1622372539713979,
        },
        "nnlm_50": {
            "16384": 1.445467433705926,
            "4096": 0.9821790346875787,
            "8192": 1.0870240876451134,
        },
        "nnlm_50_normalization": {
            "16384": 1.620640527829528,
            "4096": 1.0015304515138268,
            "8192": 1.144434040784836,
        },
        "use": {
            "16384": 19.023802764713764,
            "4096": 5.700347022153437,
            "8192": 10.494147294200957,
        },
        "use_large": {
            "16384": 58.37275729570538,
            "4096": 15.346537566557526,
            "8192": 30.610286871343853,
        },
        "xlnet": {
            "16384": 1244.9333253473044,
            "4096": 314.4162422109395,
            "8192": 625.1789539679885,
        },
        "xlnet_large": {
            "16384": 3300.903972804919,
            "4096": 826.0532148534433,
            "8192": 1649.1583985555917,
        },
    },
}
