if [ ! -d Real-Time-Voice-Cloning ]
then
    git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git
    cp preprocess_SLMIA_SR.py Real-Time-Voice-Cloning/encoder/preprocess_SLMIA_SR.py
    cp encoder_preprocess_SLMIA_SR.py Real-Time-Voice-Cloning/encoder_preprocess_SLMIA_SR.py
    cp train_SLMIA_SR.py Real-Time-Voice-Cloning/encoder/train_SLMIA_SR.py
    cp encoder_train_SLMIA_SR.py Real-Time-Voice-Cloning/encoder_train_SLMIA_SR.py
fi

cd Real-Time-Voice-Cloning

datadir=$1
dataset=$2
echo $dataset

if [ "$dataset" = "vox2" ]
then
    maxsteps=315000
elif [ "$dataset" = "ls" ]
then
    maxsteps=255000
else
    echo "dataset not supported"
    exit 1
fi

for model in "target" "shadow"
do
echo $model
preprocess_dir=$1/$dataset-preprocess/$model-member/$model-member-train
mkdir -p $preprocess_dir
echo $preprocess_dir
python encoder_preprocess_SLMIA_SR.py $1 -d $dataset-$model -o $preprocess_dir
python encoder_train_SLMIA_SR.py $dataset-$model $preprocess_dir -m saved_models -u 0 --no_visdom --max_steps $maxsteps

done

# ## train target SRS
# preprocess_dir=$1/$dataset-preprocess/target-train_speaker/target-train_speaker-train_voice
# mkdir $preprocess_dir
# python encoder_preprocess_SLMIA_SR.py $1 -d $dataset-target -o $preprocess_dir
# python encoder_train_SLMIA_SR.py $dataset-target $preprocess_dir -m saved_models -u 0 --no_visdom --max_steps $maxsteps

# ## train shadow SRS
# preprocess_dir=$1/$dataset-preprocess/shadow-train_speaker/shadow-train_speaker-train_voice
# mkdir $preprocess_dir
# python encoder_preprocess_SLMIA_SR.py $1 -d $dataset-shadow -o $preprocess_dir
# python encoder_train_SLMIA_SR.py $dataset-shadow $preprocess_dir -m saved_models -u 0 --no_visdom --max_steps $maxsteps

ln -s model/_SV2TTS/encoder ./ # torch version of encoder, will be used by model.srs; copy to the root directory to avoid import error