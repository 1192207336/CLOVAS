manual_prompts = {
    'candle': [
        ['color defect. hole. black defect. wick hole. spot. ', 'candle'],
    ],

    'capsules': [
        ['black melt. dark liquid.', 'capsules'],
        ['bubble', 'capsules'],  # 33+-->37+
    ],

    'cashew': [
        ['yellow defect. black defect. small holes. scratch. breakage. crack.', 'cashew'],
        ['small spot.', 'cashew'],

    ],

    'chewinggum': [
        ['crack. yellow defect. color defect. black defect. thread.', 'chewinggum'],
    ],

    'fryum': [
        ['broken part. purple defect. black defect. red defect. color defect. color crack. break', 'fryum'],
        # 48.4-->51.58
    ],

    'macaroni1': [
        ['broken part. red defect. black defect. crack. hole. scratch. color crack.', 'macaroni'],  # 45.66-->46.04
    ],

    'macaroni2': [
        ['broken part. red defect. black defect. crack. hole. scratch.', 'macaroni'],  # 29.12-->32.64
        ['edge. spot. chip.', 'macaroni'],
    ],

    #
    'pcb1': [  # 3.37-->8.6
        ['bent wire on pcb', 'pcb'],
        ['white scratch on pcb', 'pcb'],
        ['white defect on pcb', 'pcb'],
        ['white circle.', 'pcb']
    ],

    'pcb2': [
        ['bent wire', 'pcb'],
        ['pointed', 'pcb'],
        ['wire', 'pcb'],
    ],

    'pcb3': [
        ['bent wire on pcb', 'pcb'],
        ['scratch on pcb', 'pcb'],
        ['defect on pcb', 'pcb'],
    ],

    'pcb4': [
        ['yellow paper on pcb', 'pcb'],
        ['yellow defect on pcb', 'pcb'],
        ['black defect on pcb', 'pcb'],
        ['color defect on pcb', 'pcb'],
        ['scratch on pcb', 'pcb'],
        ['defect on pcb', 'pcb'],
    ],

    'pipe_fryum': [
        ['broken part. red defect. black defect. crack. hole. scratch.', 'fryum'],
    ],
}

ori_prompts={
        'candle': ['other', 'different colour spot', 'foreign particals on candle', 'extra wax in candle', 'weird candle wick', 'damaged corner of packaging', 'wax melded out of the candle', 'chunk of wax missing'], 
        'capsules': ['discolor', 'bubble', 'scratch', 'leak', 'misshape'], 
        'cashew': ['corner or edge breakage', 'middle breakage', 'stuck together', 'different colour spot', 'small holes', 'same colour spot', 'burnt', 'small scratches'], 
        'chewinggum': ['chunk of gum missing', 'similar colour spot', 'small cracks', 'corner missing', 'scratches'],  
        'fryum': ['corner or edge breakage', 'similar colour spot', 'other', 'middle breakage', 'stuck together', 'small cracks', 'different colour spot', 'burnt', 'fryum stuck together', 'small scratches', 'corner and edge breakage'],  
        'macaroni': ['different color spot', 'similar colour spot', 'other', 'middle breakage', 'small cracks', 'different colour spot', 'breakage down the middle', 'small chip around edge', 'chip around edge and corner', 'color spot similar to the object', 'small scratches'],  
        'pcb': ['dirt', 'missing', 'bent', 'burnt', 'melt', 'scratch', 'damage', 'extra', 'wrong place']
}
normal_prompts=[
    'a normal {}.',
    'an intact {}.',
    '{} in perfect condition.',
    'a pristine {}.',
]
gpt_prompts = {  
    'candle': {  
        'chunk of wax missing': 'a candle with a chunk of wax missing' ,
        'damaged corner of packaging': 'a candle with a damaged corner of packaging',  
        'weird candle wick': 'a candle with a weird wick',  
        'different colour spot': 'a candle with a different color spot',  
        'extra wax in candle': 'a candle with extra wax',  
        'foreign particals on candle': 'a candle with foreign particles on it',  
        'wax melded out of the candle': 'a candle with wax melded out',  
        'other': 'a candle with some other type of defect',  
    },  
    'capsules': {  
        'bubble': 'a capsule with a bubble',  
        'discolor': 'a capsule with discoloration',  
        'scratch': 'a capsule with a scratch',  
        'leak': 'a capsule with a leak',  
        'misshape': 'a misshaped capsule'  
    },  
    'cashew': {  
        'burnt': 'a burnt cashew',  
        'corner or edge breakage': 'a cashew with corner or edge breakage',  
        'middle breakage': 'a cashew with middle breakage',  
        'stuck together': 'cashews stuck together',  
        'different colour spot': 'a cashew with a different color spot',  
        'small holes': 'a cashew with small holes',  
        'same colour spot': 'a cashew with a same color spot',  
        'small scratches': 'a cashew with small scratches',
        'other': 'a cashew with some other type of defect',
    },  
    'chewinggum': {  
        'chunk of gum missing': 'a chewing gum with a chunk missing',  
        'similar colour spot': 'a chewing gum with a similar color spot',  
        'small cracks': 'a chewing gum with small cracks',  
        'corner missing': 'a chewing gum with a corner missing',  
        'scratches': 'a chewing gum with scratches',
        'other': 'a chewing gum with some other type of defect',
    },  
    'fryum': {  
        'corner or edge breakage': 'a fryum with corner or edge breakage',  
        'similar colour spot': 'a fryum with a similar color spot',  
        'other': 'a fryum with some other type of defect',  
        'middle breakage': 'a fryum with middle breakage',  
        'stuck together': 'fryums stuck together',  
        'small cracks': 'a fryum with small cracks',  
        'different colour spot': 'a fryum with a different color spot',  
        'burnt': 'a burnt fryum',  
        'fryum stuck together': 'fryums stuck together',  
        'small scratches': 'a fryum with small scratches',  
        'corner and edge breakage': 'a fryum with corner and edge breakage',
        'other': 'a fryum with some other type of defect', 
    },  
    'macaroni1': {  
        'chip around edge and corner': 'a macaroni with a chip around the edge and corner',  
        'different colour spot': 'a macaroni with a different color spot',  
        'middle breakage': 'a macaroni with middle breakage',  
        'similar colour spot': 'a macaroni with a similar color spot',  
        'small cracks': 'a macaroni with small cracks',  
        'small scratches': 'a macaroni with small scratches',
        'other': 'a macaroni with some other type of defect',
    },  
    'macaroni2': {  
        'breakage down the middle': 'a macaroni with breakage down the middle',  
        'color spot similar to the object': 'a macaroni with a color spot similar to the object',  
        'different color spot': 'a macaroni with a different color spot',  
        'small chip around edge': 'a macaroni with a small chip around the edge',  
        'small cracks': 'a macaroni with small cracks',  
        'small scratches': 'a macaroni with small scratches',
        'other': 'a macaroni with some other type of defect',  
    },  
    'pcb1': {  
        'bent': 'a bent PCB',  
        'melt': 'a PCB with melted parts', 
        'scratch': 'a PCB with scratches',  
        'missing': 'a PCB with a missing component', 
    },  
    'pcb2': {  
        'bent': 'a bent PCB',  
        'melt': 'a PCB with melted parts', 
        'scratch': 'a PCB with scratches',  
        'missing': 'a PCB with a missing component',   
    },
    'pcb3': {  
        'bent': 'a bent PCB',  
        'melt': 'a PCB with melted parts', 
        'scratch': 'a PCB with scratches',  
        'missing': 'a PCB with a missing component',   
    },
    'pcb4': {  
        'burnt': 'a burnt PCB',  
        'scratch': 'a PCB with scratches',  
        'missing': 'a PCB with a missing component',  
        'damage': 'a damaged PCB',  
        'extra': 'a PCB with extra components',  
        'wrong place': 'a PCB with components in the wrong place',
        'dirt': 'a PCB with dirt',          
    },
    'pipe_fryum': {  
        'burnt': 'a burnt fryum',  
        'corner and edge breakage': 'a fryum with corner or edge breakage',
        'different colour spot': 'a fryum with a different color spot', 
        'middle breakage': 'a fryum with middle breakage',     
        'similar colour spot': 'a fryum with a similar color spot',  
        'small scratches': 'a fryum with small scratches',  
        'stuck together': 'fryums stuck together',  
        'small cracks': 'a fryum with small cracks',  
        'other': 'a fryum with some other type of defect',  
    },  
    
}  
property_prompts = {
    'candle': 'the image of candle have 4 similar candle, with a maximum of 1 anomaly. The anomaly would not exceed 0.3 object area. ',
    'capsules': 'the image of capsule have 20 dissimilar capsule, with a maximum of 15 anomaly. The anomaly would not exceed 1. object area. ',
    'cashew': 'the image of cashew have 1 dissimilar cashew, with a maximum of 17 anomaly. The anomaly would not exceed 0.05 object area. ',
    'chewinggum': 'the image of chewinggum have 1 dissimilar chewinggum with a maximum of 50 anomaly. The anomaly would not exceed 0.2 object area. ',
    'fryum': 'the image of fryum have 1 dissimilar fryum, with a maximum of 5 anomaly. The anomaly would not exceed 1. object area. ',
    'macaroni1': 'the image of macaroni1 have 4 similar macaroni, with a maximum of 5 anomaly. The anomaly would not exceed 0.3 object area. ',
    'macaroni2': 'the image of macaroni2 have 4 dissimilar macaroni, with a maximum of 50 anomaly. The anomaly would not exceed 0.3 object area. ',
    'pcb1': 'the image of pcb1 have 1 dissimilar printed_circuit_board, with a maximum of 1 anomaly. The anomaly would not exceed 0.3 object area. ',
    'pcb2': 'the image of pcb2 have 1 dissimilar pcb, with a maximum of 9 anomaly. The anomaly would not exceed 0.3 object area. ',
    'pcb3': 'the image of pcb3 have 1 dissimilar printed_circuit_board, with a maximum of 5 anomaly. The anomaly would not exceed 0.3 object area. ',
    'pcb4': 'the image of pcb4 have 1 dissimilar pcb, with a maximum of 7 anomaly. The anomaly would not exceed 0.3 object area. ',
    'pipe_fryum': 'the image of pipe_fryum have 1 dissimilar pipe_fryum, with a maximum of 19 anomaly. The anomaly would not exceed 0.3 object area. ',
}

official_prompts = {
    "candle":
        ["damaged corner of packaging,different colour spot,other",
         "different colour spot,foreign particals on candle",
         "chunk of wax missing,damaged corner of packaging,different colour spot",
         "damaged corner of packaging,foreign particals on candle",
         "wax melded out of the candle", "damaged corner of packaging,extra wax in candle",
         "different colour spot,wax melded out of the candle",
         "weird candle wick,different colour spot",
         "damaged corner of packaging,weird candle wick",
         "chunk of wax missing,foreign particals on candle", "different colour spot",
         "damaged corner of packaging,different colour spot",
         "foreign particals on candle,wax melded out of the candle", "extra wax in candle",
         "weird candle wick", "damaged corner of packaging",
         "chunk of wax missing,different colour spot", "chunk of wax missing",
         "foreign particals on candle"
         ],
    "capsules": ["bubble,discolor,scratch", "bubble", "bubble,discolor,scratch,leak",
                 "bubble,discolor,scratch,leak,misshape", "bubble,discolor"],
    "cashew": ["stuck together", "burnt", "different colour spot,small holes",
               "corner or edge breakage", "same colour spot", "burnt,same colour spot",
               "different colour spot,same colour spot", "corner or edge breakage,small scratches",
               "burnt,different colour spot", "middle breakage,small holes", "small holes",
               "different colour spot", "burnt,corner or edge breakage", "middle breakage",
               "small scratches", "middle breakage,same colour spot"],
    "chewinggum": ["chunk of gum missing,scratches,small cracks",
                   "corner missing,similar colour spot,small cracks", "similar colour spot",
                   "corner missing,similar colour spot", "corner missing,small cracks",
                   "corner missing", "chunk of gum missing,scratches",
                   "chunk of gum missing,small cracks", "chunk of gum missing",
                   "chunk of gum missing,corner missing", "scratches",
                   "scratches,similar colour spot,small cracks",
                   "chunk of gum missing,corner missing,small cracks", "scratches,similar colour spot",
                   "corner missing,scratches"],
    "fryum": ["middle breakage,small scratches", "middle breakage,similar colour spot",
              "similar colour spot,small scratches", "burnt", "similar colour spot",
              "different colour spot", "fryum stuck together",
              "different colour spot,similar colour spot", "corner or edge breakage",
              "corner or edge breakage,small scratches", "middle breakage", "similar colour spot,other",
              "small scratches"],
    "macaroni1": ["chip around edge and corner,small cracks", "middle breakage,small scratches",
                  "chip around edge and corner", "small cracks", "small cracks,small scratches",
                  "different colour spot", "similar colour spot",
                  "different colour spot,similar colour spot",
                  "chip around edge and corner,small scratches", "small scratches"],
    "macaroni2": ["small chip around edge", "small cracks",
                  "breakage down the middle,color spot similar to the object,different color spot,small chip around edge,small cracks",
                  "small cracks,small scratches", "small scratches",
                  "breakage down the middle,color spot similar to the object,different color spot,small chip around edge",
                  "breakage down the middle,color spot similar to the object,different color spot,small chip around edge,small cracks,small scratches,other",
                  "different color spot", "breakage down the middle,small scratches",
                  "breakage down the middle", "color spot similar to the object",
                  "breakage down the middle,color spot similar to the object,different color spot,small chip around edge,small cracks,small scratches",
                  "small chip around edge,small scratches"],
    "pcb1": ["melt,scratch", "scratch,missing", "bent", "missing", "melt,missing", "melt",
             "melt,scratch,missing", "scratch", "bent,melt"],
    "pcb2": ["melt,scratch", "scratch,missing", "bent", "missing", "melt", "scratch", "bent,melt"],
    "pcb3": ["scratch,missing", "bent", "missing", "melt", "melt,missing", "melt,scratch",
             "melt,scratch,missing", "scratch", "bent,melt"],
    "pcb4": ["burnt,dirt", "missing,dirt", "burnt,scratch,dirt", "scratch,extra,dirt", "burnt",
             "damage,dirt", "wrong place", "burnt,extra", "missing,damage,extra", "scratch,damage,dirt",
             "extra", "scratch,damage", "scratch,missing", "scratch,missing,dirt", "wrong place,dirt",
             "extra,dirt", "missing,wrong place", "missing,damage", "scratch,damage,extra", "missing",
             "damage", "damage,extra", "scratch,extra,wrong place", "scratch,dirt", "scratch,extra",
             "scratch"],
    "pipe_fryum": ["middle breakage,small scratches,small cracks",
                   "similar colour spot,small scratches", "stuck together",
                   "burnt", "different colour spot", "similar colour spot",
                   "corner and edge breakage", "burnt,small scratches",
                   "small scratches", "middle breakage,small cracks"]}
