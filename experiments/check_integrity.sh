echo "Checking MNIST"
for method in "filter" "gibbs-eb-ef" "pmcmc" "twisted"; do
    for nparticles in "10" "100"; do
        n1=$(find ./imgs/results_inpainting/arrs/mnist-15-lin-"$nparticles"-*-"$method"* | wc -l)
        n2=$(find ./imgs/results_supr/arrs/mnist-4-lin-"$nparticles"-*-"$method"* | wc -l)
        echo "Checking $method with $nparticles particles: inpainting $n1 supr $n2"
    done
done

echo "Checking CELABA"
for method in "filter" "gibbs-eb-ef" "pmcmc" "twisted"; do
    for nparticles in "2" "10"; do
        n1=$(find ./imgs/results_inpainting/arrs/celeba-64-32-lin-"$nparticles"-*-"$method"* | wc -l)
        n2=$(find ./imgs/results_supr/arrs/celeba-64-2-lin-"$nparticles"-*-"$method"* | wc -l)
        echo "Checking $method with $nparticles particles: inpainting $n1 supr $n2"
    done
done
