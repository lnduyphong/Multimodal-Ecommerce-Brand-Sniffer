# TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')

# PERCENTAGE=80

# MEM_LIMIT=$(( TOTAL_MEM * PERCENTAGE / 100 ))

# ulimit -v $MEM_LIMIT

nohup python demo.py > /home/lnduyphong/ecom/Result/output.log