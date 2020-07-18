#include <iostream>
#include <math.h>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <limits>
#include <random>
#include <fstream>

struct Datapoint
{
    Datapoint() = default;

    Datapoint(const int n_feats_): n_feats(n_feats_)
    {
        features.resize(n_feats);
    }

    Datapoint(const std::vector<float> &feats_): n_feats(feats_.size()), features(feats_) {};

    Datapoint operator-(const Datapoint &other) const
    {
        if (other.size() != features.size())
        {
            std::cerr << "Datapoint::operator- : Size doesn't match!\n";
            return Datapoint();
        }
        int n_feats = other.size();
        Datapoint res(n_feats);

        for (int i = 0; i < n_feats; ++i)
        {
            res.features[i] = features[i] - other.features[i];
        }

        return res;
    }

    Datapoint operator+(const Datapoint &other) const
    {
        if (other.size() != features.size())
        {
            std::cerr << "Datapoint::operator+ : Size doesn't match!\n";
            return Datapoint();
        }
        int n_feats = other.size();
        Datapoint res(n_feats);

        for (int i = 0; i < n_feats; ++i)
        {
            res.features[i] = features[i] + other.features[i];
        }

        return res;
    }

    int size() const
    {
        return n_feats;
    }

    // Member variables
    std::vector<float> features;
    int n_feats;
};

using Dataset = std::vector<Datapoint>;

float square (float val)
{
    return val * val;
}

float get_L2_dist (const Datapoint &dp1, const Datapoint &dp2)
{
    float sq_dist = 0;
    auto diff = dp1 - dp2;
    for (const auto f : diff.features)
    {
        sq_dist += square(f);
    }

    return sq_dist;
}


class KMeans
{
public:
    KMeans(int k): k_(k), max_iter_(100), loss_tol_(1e-5)
    {
        centroids_.reserve(k_);
        clusters_.resize(k_);
    }

    void assign_clusters (const Dataset &data)
    {
        // Reset clusters
        clusters_ = std::vector<Dataset>(k_);

        for (const auto &pt : data)
        {
            float min_sq_dist = std::numeric_limits<float>::max();
            int cluster_id = 0;

            for (int id = 0; id < k_; ++id)
            {
                auto &centroid = centroids_[id];
                float sq_dist = get_L2_dist(pt, centroid);
                if (sq_dist < min_sq_dist)
                {
                    min_sq_dist = sq_dist;
                    cluster_id = id;
                }
            }

            clusters_[cluster_id].push_back(pt);
        }

        return;
    }

    void update_centroids(int n_feats)
    {
        for (int id = 0; id < k_; ++id)
        {
            Dataset &cluster_pts = clusters_[id];
            int n_pts = cluster_pts.size();

            // If the cluster is empty, use the previous centroid.
            if (n_pts == 0) 
            {
                continue;
            }

            Datapoint centroid (n_feats);

            // Sum each feature
            for (const auto &pt : cluster_pts)
            {
                centroid = centroid + pt;
            }

            // Take average
            for (auto &feat : centroid.features)
            {
                feat /= n_pts;
            }

            centroids_[id] = centroid;
        }

        return;
    }

    float get_loss(int n_data)
    {
        float loss = 0;

        for (int id = 0; id < k_; ++id)
        {
            auto &centroid = centroids_[id];
            for (const auto &pt : clusters_[id])
            {
                loss += get_L2_dist(centroid, pt);
            }
        }

        return loss / n_data;
    }

    void fit(const Dataset &data)
    {
        int n_data  = data.size();
        std::cout << "Number of data: " << n_data << std::endl;

        if (n_data == 0)
        {
            std::cerr << "Input data is empty!\n";
            return;
        }

        int n_feats = data.front().size();
        std::cout << "Number of features: " << n_feats << std::endl;

        // Initialization - reset
        centroids_.clear();
        loss_history_.clear();

        // Initialize centroids
        std::cout << "Initializing\n";
        for (int i = 0; i < k_; ++i)
        {
            int rand_ind = rand() % k_;
            centroids_.push_back(data[rand_ind]);
        }

        // Iteratively udpate the model
        for (int n_iter = 0; n_iter < max_iter_; ++n_iter)
        {
            // Step 1: assign cluster id to each data point
            std::cout << "Step1\n";
            assign_clusters(data);

            // Step 2: update centroids
            std::cout << "Step2\n";
            update_centroids(n_feats);

            // Check the loss
            std::cout << "Loss check\n";
            float cur_loss = get_loss(n_data);
            std::cout << "Iteration " << n_iter << ", loss: " << cur_loss << std::endl;
            
            if (loss_history_.size() > 0 && fabs(loss_history_.back() - cur_loss) < loss_tol_)
            {
                std::cout << "Converge at iteration-" << n_iter << std::endl;
                return;
            }
            
            loss_history_.push_back(cur_loss);
        }

        return;
    }

    std::vector<Dataset>& get_clusters()
    {
        return clusters_;
    }

private:
    int k_;
    int max_iter_;
    float loss_tol_;
    Dataset centroids_;
    std::vector<Dataset> clusters_;
    std::vector<float> loss_history_;
};


int main(int argc, const char* argv[]) 
{
    std::random_device rnd_device;
    std::mt19937 gen {rnd_device()};  // Generates random integers
    std::normal_distribution<> dist {0.0, 10};

    auto normal_gen = [&dist, &gen](){ return dist(gen); };

    int n_data = 1000;
    int n_feats = 2;
    int k = 2;

    Datapoint center1 (std::vector<float>({10, 10}));
    Datapoint center2 (std::vector<float>({-10, -10}));
    Dataset data, cluster1_true, cluster2_true;
    int n_cluster1 = n_data / 2;
    int n_cluster2 = n_data / 2;

    data.reserve(n_data);
    cluster1_true.reserve(n_cluster1);
    cluster2_true.reserve(n_cluster2);

    // Generate random dataset with 2 uniform distributions centering at (10, 10) and (-10, -10), respectively
    for (int i = 0; i < n_cluster1; ++i)
    {
        std::vector<float> feat(2);
        std::generate(feat.begin(), feat.end(), normal_gen);
        Datapoint pt (feat);
        pt = pt + center1;
        data.push_back(pt);
        cluster1_true.push_back(pt);
    }

    for (int i = 0; i < n_cluster2; ++i)
    {
        std::vector<float> feat(2);
        std::generate(feat.begin(), feat.end(), normal_gen);
        Datapoint pt (feat);
        pt = pt + center2;
        data.push_back(pt);
        cluster2_true.push_back(pt);
    }

    std::random_shuffle(data.begin(), data.end());

    KMeans kmeans(2);

    kmeans.fit(data);

    auto clusters = kmeans.get_clusters();


    // Output data set to csv file
    std::ofstream output_file;
    output_file.open("result.csv");
    output_file << "cluster1_true;;cluster2_true;;cluster1;;cluster2\n";

    int n_pred_cluster1 = clusters[0].size();
    int n_pred_cluster2 = clusters[1].size();
    int ind_cluster1_true = 0;
    int ind_cluster2_true = 0;
    int ind_cluster1 = 0;
    int ind_cluster2 = 0;

    while (ind_cluster1_true < n_cluster1 || ind_cluster1 < n_pred_cluster1 ||
           ind_cluster2_true < n_cluster2 || ind_cluster2 < n_pred_cluster2)
    {
        if (ind_cluster1_true < n_cluster1)
        {
            auto &pt = cluster1_true[ind_cluster1_true++];
            output_file << pt.features[0] << ";" << pt.features[1] << ";";
        }
        else
        {
            output_file << ";;";
        }

        if (ind_cluster2_true < n_cluster2)
        {
            auto &pt = cluster2_true[ind_cluster2_true++];
            output_file << pt.features[0] << ";" << pt.features[1] << ";";
        }
        else
        {
            output_file << ";;";
        }

        if (ind_cluster1 < n_pred_cluster1)
        {
            auto &pt = clusters[0][ind_cluster1++];
            output_file << pt.features[0] << ";" << pt.features[1] << ";";
        }
        else
        {
            output_file << ";;";
        }

        if (ind_cluster2 < n_pred_cluster2)
        {
            auto &pt = clusters[1][ind_cluster2++];
            output_file << pt.features[0] << ";" << pt.features[1];
        }
        
        output_file << std::endl;
    }

    output_file.close();


    return 0;
}