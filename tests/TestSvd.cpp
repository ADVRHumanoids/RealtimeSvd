#include <gtest/gtest.h>
#include <lapack_svd/lapack_svd.h>
#include <algorithm>
#include <catch_malloc.h>

class TestSvd: public ::testing::Test {
    

protected:

     TestSvd(){
         
     }

     virtual ~TestSvd() {
     }

     virtual void SetUp() {
         
     }

     virtual void TearDown() {
     }
     
     
};

TEST_F(TestSvd, checkVsEigen)
{
    const int n = 3, m = 7;
    
    Eigen::MatrixXd K(n, m);
    LapackSvd svd(K.rows(), K.cols());
    
    for(int i = 0; i < 3; i++)
    {
    
        K.setRandom(n, m);
        
        ASSERT_TRUE(svd.compute(K));
        
        Eigen::MatrixXd Khat = svd.matrixU()*svd.singularValues().asDiagonal()*svd.matrixV().transpose().topRows(K.rows());
        
        std::cout << "************\n" << svd.matrixU() << "\n" << std::endl;
        std::cout << svd.singularValues().transpose() << "\n" << std::endl;
        std::cout << svd.matrixV() << "\n" << std::endl;
        std::cout << K << "\n" << std::endl;
        std::cout << Khat << "\n" << std::endl;
        
        
        
        EXPECT_NEAR( (K - Khat).norm(), 0.0, 1e-6 );
        
        Eigen::JacobiSVD<Eigen::MatrixXd> eig_svd(K, Eigen::ComputeFullU|Eigen::ComputeFullV);
        
        EXPECT_NEAR( (svd.singularValues() - eig_svd.singularValues()).norm(), 0.0, 1e-6 );
        
    }
    
};


TEST_F(TestSvd, checkVsEigenBig)
{
    const int n = 6, m = 30;
    
    Eigen::MatrixXd K(n, m);
    LapackSvd svd(K.rows(), K.cols());
    
    for(int i = 0; i < 3; i++)
    {
    
        K.setRandom(n, m);
        
        ASSERT_TRUE(svd.compute(K));
        
        Eigen::MatrixXd Khat = svd.matrixU()*svd.singularValues().asDiagonal()*svd.matrixV().transpose().topRows(K.rows());
        
        std::cout << "************\n" << svd.matrixU() << "\n" << std::endl;
        std::cout << svd.singularValues().transpose() << "\n" << std::endl;
        std::cout << svd.matrixV() << "\n" << std::endl;
        std::cout << K << "\n" << std::endl;
        std::cout << Khat << "\n" << std::endl;
        
        
        
        EXPECT_NEAR( (K - Khat).norm(), 0.0, 1e-6 );
        
        Eigen::JacobiSVD<Eigen::MatrixXd> eig_svd(K, Eigen::ComputeFullU|Eigen::ComputeFullV);
        
        EXPECT_NEAR( (svd.singularValues() - eig_svd.singularValues()).norm(), 0.0, 1e-6 );
        
    }
    
};

TEST_F(TestSvd, checkVsEigen1)
{
    const int n = 7, m = 3;
    
    Eigen::MatrixXd K(n, m);
    LapackSvd svd(K.rows(), K.cols());
    
    for(int i = 0; i < 3; i++)
    {
    
        K.setRandom(n, m);
        
        ASSERT_TRUE(svd.compute(K));
        
        Eigen::MatrixXd Khat = svd.matrixU().leftCols(m)*svd.singularValues().asDiagonal()*svd.matrixV().transpose();
        
        std::cout << "************\n" << svd.matrixU() << "\n" << std::endl;
        std::cout << svd.singularValues().transpose() << "\n" << std::endl;
        std::cout << svd.matrixV() << "\n" << std::endl;
        std::cout << K << "\n" << std::endl;
        std::cout << Khat << "\n" << std::endl;
        
        
        
        EXPECT_NEAR( (K - Khat).norm(), 0.0, 1e-6 );
        
        Eigen::JacobiSVD<Eigen::MatrixXd> eig_svd(K, Eigen::ComputeFullU|Eigen::ComputeFullV);
        
        EXPECT_NEAR( (svd.singularValues() - eig_svd.singularValues()).norm(), 0.0, 1e-6 );
        
    }
    
};


TEST_F(TestSvd, checkMalloc)
{
    const int n = 6, m = 30;
    
    Eigen::MatrixXd K(n, m);
    LapackSvd svd(K.rows(), K.cols());
    
//     XBot::Utils::MallocFinder::SetThrowOnMalloc(true);
//     XBot::Utils::MallocFinder::SetThrowOnFree(true);
    
    XBot::Utils::MallocFinder::SetOnMalloc([](){ASSERT_TRUE(false && "malloc");});
    XBot::Utils::MallocFinder::SetOnFree([](){ASSERT_TRUE(false && "free");});
    
    for(int i = 0; i < 3; i++)
    {
    
        K.setRandom(n, m);
        
        XBot::Utils::MallocFinder::Enable();
        ASSERT_TRUE(svd.compute(K));
        XBot::Utils::MallocFinder::Disable();
        
        Eigen::MatrixXd Khat = svd.matrixU()*svd.singularValues().asDiagonal()*svd.matrixV().transpose().topRows(K.rows());
        
        std::cout << "************\n" << svd.matrixU() << "\n" << std::endl;
        std::cout << svd.singularValues().transpose() << "\n" << std::endl;
        std::cout << svd.matrixV() << "\n" << std::endl;
        std::cout << K << "\n" << std::endl;
        std::cout << Khat << "\n" << std::endl;
        
        
        
        EXPECT_NEAR( (K - Khat).norm(), 0.0, 1e-6 );
        
        Eigen::JacobiSVD<Eigen::MatrixXd> eig_svd(K, Eigen::ComputeFullU|Eigen::ComputeFullV);
        
        EXPECT_NEAR( (svd.singularValues() - eig_svd.singularValues()).norm(), 0.0, 1e-6 );
        
    }
    
//     XBot::Utils::MallocFinder::SetThrowOnMalloc(false);
//     XBot::Utils::MallocFinder::SetThrowOnFree(false);
    
};

TEST_F(TestSvd, checkVsEigenRank)
{
    const int n = 7, m = 3;
    
    Eigen::MatrixXd K(n, m);
    LapackSvd svd(K.rows(), K.cols());
    
    for(int i = 0; i < 3; i++)
    {
    
        K.setRandom(n, m);
        K.col(1) = K.col(2);
        
        ASSERT_TRUE(svd.compute(K));
        
        std::cout << "************\n" << svd.singularValues().transpose() << "\n" << std::endl;
        std::cout << "Rank = " << svd.rank() << "\n" << std::endl;
        
        Eigen::JacobiSVD<Eigen::MatrixXd> eig_svd(K, Eigen::ComputeFullU|Eigen::ComputeFullV);
        
        ASSERT_EQ(eig_svd.rank(), svd.rank());
    }
    
};

TEST_F(TestSvd, checkVsEigenSolve)
{
    const int n = 7, m = 3;
    
    Eigen::MatrixXd K(n, m);
    Eigen::VectorXd b(n);
    Eigen::VectorXd x(m);
    LapackSvd svd(K.rows(), K.cols());
    
    for(int i = 0; i < 3; i++)
    {
    
        K.setRandom(n, m);
        b.setRandom(n);
        
        ASSERT_TRUE(svd.compute(K));
        
        Eigen::JacobiSVD<Eigen::MatrixXd> eig_svd(K, Eigen::ComputeFullU|Eigen::ComputeFullV);
        
        svd.solve(b,x);
        std::cout << "************\nLapack SVD: " << x << "\n" << std::endl;
        std::cout << "************\nEigen SVD: " << eig_svd.solve(b) << "\n" << std::endl;
        
        EXPECT_NEAR( (eig_svd.solve(b) - x).norm(), 0.0, 1e-6 );
    }
    
};


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}