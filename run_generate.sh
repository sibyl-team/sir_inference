#!/bin/bash

for s in {13..20}
do
	python generate_trans.py -s $s &
done

