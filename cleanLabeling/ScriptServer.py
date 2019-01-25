from ComputeMetricsLPD import ComputeMetricsLPD
path='/home/ftamagnan/dataset/lpd_5/lpd_5_cleansed/'
path_tags= [
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_Blues.id',
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_Country.id',
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_Electronic.id',
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_Folk.id',
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_Jazz.id',
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_Latin.id',
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_Metal.id',
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_New-Age.id',
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_Pop.id', # 8
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_Punk.id',
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_Rap.id',
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_Reggae.id',
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_RnB.id',
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_Rock.id', # 13
    '/home/herman/lpd/id_lists/tagtraum/tagtraum_World.id',
]
cm= ComputeMetricsLPD(path,path_tags)
cm.macro_iteration()