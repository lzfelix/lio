source activate hyper2

for fn in 'sphere' 'csendes' 'salomon' 'ackley1' 'alpine1' 'rastrigin' 'schwefel' 'brown'
do
    for vars in 10 25 50 100
    do
        python run_experiment.py $fn -n_vars $vars -result_dest ./results/$fn/$vars
    done
done

