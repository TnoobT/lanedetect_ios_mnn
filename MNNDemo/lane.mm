#include "lane.h"

bool LaneDetect::hasGPU = false;
bool LaneDetect::toUseGPU = false;
LaneDetect *LaneDetect::detector = nullptr;

LaneDetect::LaneDetect(const std::string &mnn_path, bool useGPU)
{
    toUseGPU = hasGPU && useGPU;

    NSString *path = [NSString stringWithCString:"mlsd_with_max_sigmoid.mnn" encoding:[NSString defaultCStringEncoding]];
    auto model_path = [[NSBundle mainBundle] pathForResource:path ofType:nil];
    if (model_path.length <= 0) {
        NSLog(@"model path is nil");
        return;
    }
    
    
    m_net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile([model_path UTF8String]));
    m_backend_config.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;  // 精度
    m_backend_config.power = MNN::BackendConfig::Power_Normal; // 功耗
    m_backend_config.memory = MNN::BackendConfig::Memory_Normal; // 内存占用
    m_config.backendConfig = &m_backend_config;
    m_config.numThread = 4;
    if (useGPU) {
        m_config.type = MNN_FORWARD_OPENCL;
    }
    m_config.backupType = MNN_FORWARD_CPU;

    MNN::CV::ImageProcess::Config img_config; // 图像处理
    ::memcpy(img_config.mean, m_mean_vals, sizeof(m_mean_vals)); // (img - mean)*norm
    ::memcpy(img_config.normal, m_norm_vals, sizeof(m_norm_vals));
    img_config.sourceFormat = MNN::CV::BGR;
    img_config.destFormat = MNN::CV::RGB;
    pretreat = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(img_config));
    MNN::CV::Matrix trans;
    trans.setScale(1.0f, 1.0f); // scale
    pretreat->setMatrix(trans);

    m_session = m_net->createSession(m_config); //创建session
    m_inTensor = m_net->getSessionInput(m_session, nullptr);
    m_net->resizeTensor(m_inTensor, {1, 3, m_input_size, m_input_size});
    m_net->resizeSession(m_session);
    //std::cout << "session created" << std::endl;
}


LaneDetect::~LaneDetect()
{
    m_net->releaseModel();
    m_net->releaseSession(m_session);
}

inline int LaneDetect::clip(float value)
{
    if (value > 0 && value < m_input_size)
        return int(value);
    else if (value < 0)
        return 1;
    else
        return m_input_size - 1;

}


std::vector<LaneDetect::Lanes> LaneDetect::decodeHeatmap(const float* hm,int w, int h, double threshold, double lens_threshold)
{
    // 线段中心点(256*256),线段偏移(4*256*256)
    const float*  displacement = hm+m_hm_size*m_hm_size;
    // exp(center,center);
    std::vector<float> center;
    for (int i = 0;i < m_hm_size*m_hm_size; i++)
    {
        center.push_back( 1/(exp(-hm[i]) + 1) ); // mlsd.mnn原始需要1/(exp(-hm[i]) + 1)
    }
    center.resize(m_hm_size*m_hm_size);

    std::vector<int> index(center.size(), 0);
    for (int i = 0 ; i != index.size() ; i++) {
        index[i] = i;
    }
    sort(index.begin(), index.end(),
        [&](const int& a, const int& b) {
            return (center[a] > center[b]); // 从大到小排序
        }
    );
    std::vector<Lanes> lanes;
    
    for (int i = 0; i < index.size(); i++)
    {
        int yy = index[i]/m_hm_size; // 除以宽得行号
        int xx = index[i]%m_hm_size; // 取余宽得列号
        Lanes Lane;
        Lane.x1 = xx + displacement[index[i] + 0*m_hm_size*m_hm_size];
        Lane.y1 = yy + displacement[index[i] + 1*m_hm_size*m_hm_size];
        Lane.x2 = xx + displacement[index[i] + 2*m_hm_size*m_hm_size];
        Lane.y2 = yy + displacement[index[i] + 3*m_hm_size*m_hm_size];
        Lane.lens = sqrt(pow(Lane.x1 - Lane.x2,2) + pow(Lane.y1 - Lane.y2,2));
        Lane.conf = center[index[i]];

        if (center[index[i]] > threshold && lanes.size() < m_top_k)
        {
            if ( Lane.lens > lens_threshold)
            {
                Lane.x1 = clip(w * Lane.x1 / (m_input_size / 2));
                Lane.x2 = clip(w * Lane.x2 / (m_input_size / 2));
                Lane.y1 = clip(h * Lane.y1 / (m_input_size / 2));
                Lane.y2 = clip(h * Lane.y2 / (m_input_size / 2));
                lanes.push_back(Lane);
            }
        }
        else
            break;
    }
    
    return lanes;

}


std::vector<LaneDetect::Lanes> LaneDetect::detect(UIImage *image, double threshold, double lens_threshold)
{
    // 图像处理
    int width = image.size.width;
    int height = image.size.height;
    auto imageSource = utility::UIImageGetData(image);
    if (width == 333)
        printf("111");
    
    cv::Mat preImage(m_input_size, m_input_size, CV_8UC3);
    cv::Mat image_input(height, width, CV_8UC4, imageSource.get());
    cv::cvtColor(image_input, image_input, cv::COLOR_RGBA2BGR); // 四通道需要转换！
    cv::resize(image_input, preImage, cv::Size(m_input_size, m_input_size));
    int b = image_input.channels();
    int a = preImage.channels();
    pretreat->convert(preImage.data, m_input_size, m_input_size, 0, m_inTensor);
    // 推理
    m_net->runSession(m_session);
    MNN::Tensor *output= m_net->getSessionOutput(m_session, NULL);
    
    MNN::Tensor tensor_scores_host(output, output->getDimensionType());
    output->copyToHostTensor(&tensor_scores_host);

    auto score = output->host<float>(); // 得到结果指针
    std::vector<LaneDetect::Lanes> lanes = decodeHeatmap(score, width, height, threshold,lens_threshold);
   
    
    return lanes;
}
