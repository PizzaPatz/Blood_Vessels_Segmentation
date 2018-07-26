import matplotlib.pyplot as plt
import sys
#====================================#
# Check for correct arguments
# 1 = architecture
# 2 = start interval
# 3 = end interval
try:
	print('[1] Repeating argument(s)\n')
	print('Architecutre: '+sys.argv[1]+'\n') # baseline
	print('Start interval: '+sys.argv[2])
	print('End interval: '+sys.argv[3])

	architecture = sys.argv[1]
	start_interval = int(sys.argv[2])
	end_interval = int(sys.argv[3])

	if architecture in ['baseline'] and \
		start_interval < 151 and end_interval < 151 and \
		start_interval < end_interval:
		print('[2] Starting process of graphing')
	else:
		print('[2] Invalid inputs')
		sys.exit(0)
except IndexError:
	print('[1] Invalid number of argument(s)')
	sys.exit(0)

#====================================#

# ======================================================= #

out_acc = architecture+'_net_acc_'+str(end_interval)+'_epoch_32'
out_loss = architecture+'_net_loss_'+str(end_interval)+'_epoch_32'

# ======================================================= #

plt.switch_backend('agg')

with open('../batch_size_32/checkpoint_training/history/'+architecture+'/all_acc.txt') as f:
    all_accuracy = f.readlines()
all_accuracy = [x.strip() for x in all_accuracy]

with open('../batch_size_32/checkpoint_training/history/'+architecture+'/all_loss.txt') as f:
    all_loss = f.readlines()
all_loss = [x.strip() for x in all_loss]

with open('../batch_size_32/checkpoint_training/history/'+architecture+'/all_acc_val.txt') as f:
    all_val_accuracy = f.readlines()
all_val_accuracy = [x.strip() for x in all_val_accuracy]

with open('../batch_size_32/checkpoint_training/history/'+architecture+'/all_loss_val.txt') as f:
    all_val_loss = f.readlines()
all_val_loss = [x.strip() for x in all_val_loss]

plt.style.use("ggplot")

x = range(start_interval,end_interval+1)
all_accuracy = all_accuracy[start_interval-1:end_interval]
all_loss = all_loss[start_interval-1:end_interval]
all_val_accuracy = all_val_accuracy[start_interval-1:end_interval]
all_val_loss = all_val_loss[start_interval-1:end_interval]

if architecture == 'baseline':
	architecture_name = 'Baseline'

plt.figure()
plt.plot(x, all_accuracy, label="Accuracy")
plt.plot(x, all_val_accuracy, label="Validation Accuracy")
plt.title(architecture_name+" Accuracy")
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy")
#===== Set limit
plt.ylim(0.93,0.97)
#====

plt.legend(loc="lower right")
plt.savefig('../graphs/'+out_acc)
plt.close()

plt.figure()
plt.plot(x, all_loss, label="Loss")
plt.plot(x, all_val_loss, label="Validation Loss")
plt.title(architecture_name+" Loss")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
#==== Set limit
plt.ylim(0.075,0.425)
#====
plt.legend(loc="upper right")
plt.savefig('../graphs/'+out_loss)
plt.close()
