import configparser as cp

def count_lines(filename, chunk_size=1<<13):
    with open(filename) as file:
        return sum(chunk.count('\n')
                   for chunk in iter(lambda: file.read(chunk_size), ''))

def count_lines1 (filename) :
    #slower than count_lines
    return sum(1 for line in open(filename))

def get_dir_for_model (model) :
    conffile = 'globals.conf'
    config = cp.ConfigParser(inline_comment_prefixes="#")
    config.read (conffile) 
    print ("conffile="+conffile)
    configactual = config[model]
    
    print("dir with data: "+configactual['outdir'])
    return configactual.get('outdir')
                       
