/Applications/spark-1.6.0-bin-hadoop2.6/ec2/spark-ec2 \
--key-pair=joojayhuyn \
--identity-file=/Users/joojayhuyn/.ssh/joojayhuyn.pem \
--region=us-west-2 \
--instance-type=m4.xlarge \
--slaves=4 \
launch joojayhuyn-spark-cluster