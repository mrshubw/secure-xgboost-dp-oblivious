# source ../.venv/bin/activate

# ./build_project.sh
# python predict.py -d higgs -t 5
# python predict.py -d higgs -t 10
# python predict.py -d higgs -t 20
# python predict.py -d higgs -t 40
# python predict.py -d allstate -t 5
# python predict.py -d allstate -t 10
# python predict.py -d allstate -t 20
# python predict.py -d allstate -t 40
# python predict.py -d covtype -t 5
# python predict.py -d covtype -t 10
# python predict.py -d covtype -t 20
# python predict.py -d covtype -t 40

# ./build_project.sh --O
# python predict.py -d higgs -t 5
# python predict.py -d higgs -t 10
# python predict.py -d higgs -t 20
# python predict.py -d higgs -t 40
# python predict.py -d allstate -t 5
# python predict.py -d allstate -t 10
# python predict.py -d allstate -t 20
# python predict.py -d allstate -t 40
# python predict.py -d covtype -t 5
# python predict.py -d covtype -t 10
# python predict.py -d covtype -t 20
# python predict.py -d covtype -t 40

./build_project.sh --DO
# python predict.py -d higgs -t 5
# python predict.py -d higgs -t 10
# python predict.py -d higgs -t 20
# python predict.py -d higgs -t 40
# python predict.py -d allstate -t 5
# python predict.py -d allstate -t 10
# python predict.py -d allstate -t 20
# python predict.py -d allstate -t 40
python predict.py -d covtype -t 5
python predict.py -d covtype -t 10
python predict.py -d covtype -t 20
python predict.py -d covtype -t 40



# # source ../.venv/bin/activate

# # datasets=("higgs" "allstate" "covtype")
# datasets=("covtype")
# treesnums=(5 10 20 40)
# # treesnums=(40)

# # ./build_project.sh
# # for dataset in "${datasets[@]}"
# # do
# #     for treesnum in "${treesnums[@]}"
# #     do
# #         for depth in {2..10}
# #         do
# #             python predict.py -d $dataset -t $treesnum -D $depth
# #         done
# #     done
# # done

# # ./build_project.sh --O
# # for dataset in "${datasets[@]}"
# # do
# #     for treesnum in "${treesnums[@]}"
# #     do
# #         for depth in {2..10}
# #         do
# #             python predict.py -d $dataset -t $treesnum -D $depth
# #         done
# #     done
# # done

# # ./build_project.sh --DO
# for dataset in "${datasets[@]}"
# do
#     for treesnum in "${treesnums[@]}"
#     do
#         for depth in {8..10}
#         do
#             python predict.py -d $dataset -t $treesnum -D $depth
#         done
#     done
# done
