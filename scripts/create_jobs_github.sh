#!/bin/tcsh

#set the PATH
set PATH = ./

foreach dataset (TaFeng TMall sTMall Gowalla)
    #please fix decay as 1.0 when model is not SNBR
    foreach decay (0.2 0.4 0.6 0.8 1.0)
        foreach l2 (1e-5 1e-4 1e-3 1e-2 1e-1)
            foreach dim (16 32 64 128)
                #we don't use this parameter
                foreach k (0)
                    foreach testOrder (1 2 3)
                        foreach model ($2)
                            if ($model == FREQ) then
                                set jobPath = $PATH/$1/$model\_$dataset\_$model\_$decay\_$l2\_$testOrder.job
                            else 
                                set jobPath = $PATH/$1/$model\_$dataset\_$model\_$decay\_$l2\_$dim\_$k\_$testOrder.job
                            endif
                            rm $jobPath

                            if ($model == FPMC) then
                                set batchSize=500
                            else
                                set batchSize=100
                            endif

                            if (($model == FPMC || $model == Dream) && $dataset != sTMall) then
                                set numIter=300
                            else
                                set numIter=100
                            endif

                            echo python main.py --dataset=$dataset --decay=$decay --l2=$l2 --dim=$dim --numIter=$numIter --model=$model --isTrain=$3 --k=$k --testOrder=$testOrder --isPreTrain=0 --batchSize=$batchSize --mode='time_split' >> $jobPath
                        end
                    end
                end
            end
        end
    end
end
