##----------------------------------------------------
## Created by Ivan Luiz de Oliveira
## Unesp Sorocaba Aug. 2019
##----------------------------------------------------

config=open('config.cfg','r')
config_data=config.read()
config.close()
config_data=config_data.split('::')
config_data=config_data[1].split('\n')
for i in range(1,len(config_data)):
    config_data[i] = config_data[i].split('==')
enable_startup_run = int(config_data[14][1])

if enable_startup_run==1:
    exec(open("Operation.py").read())
else:
    print('Startup run of "Operation.py" is disabled...')