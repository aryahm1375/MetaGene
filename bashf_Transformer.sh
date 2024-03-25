for global_bath in 32 64; do
    for gcn in 32 64 128; do
        for alpha in 0.1 0.3 0.5 0.7 0.9 1; do
          for target in 255 304 771 830; do
                python new_meta_learning_Transformer.py $global_bath $gcn $alpha $target
          done
        done
    done
done

