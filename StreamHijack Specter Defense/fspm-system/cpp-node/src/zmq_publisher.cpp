#include "zmq_publisher.hpp"
#include <cstring>
#include <ctime>
#include <stdexcept>

namespace fspm {

// Simple binary pack: node_id_len(2) + node_id + score(8) + threshold(8)
//                     + n_feat(4) + features(n*4) + ts_ns(8)
static std::vector<uint8_t> pack_anomaly(const std::string& node_id,
                                          double score, double threshold,
                                          const std::vector<float>& feats) {
  uint16_t id_len = static_cast<uint16_t>(node_id.size());
  uint32_t n_feat = static_cast<uint32_t>(feats.size());
  size_t   total  = 2 + id_len + 8 + 8 + 4 + n_feat * 4 + 8;
  std::vector<uint8_t> buf(total);
  uint8_t* p = buf.data();

  std::memcpy(p, &id_len, 2);  p += 2;
  std::memcpy(p, node_id.data(), id_len); p += id_len;
  std::memcpy(p, &score, 8);   p += 8;
  std::memcpy(p, &threshold, 8); p += 8;
  std::memcpy(p, &n_feat, 4);  p += 4;
  std::memcpy(p, feats.data(), n_feat * 4); p += n_feat * 4;

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  int64_t ns = static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
  std::memcpy(p, &ns, 8);

  return buf;
}

ZmqPublisher::ZmqPublisher(const std::string& endpoint)
  : context_(1), socket_(context_, zmq::socket_type::pub),
    topic_("FSPM_ANOMALY")
{
  socket_.connect(endpoint);
}

ZmqPublisher::~ZmqPublisher() {
  socket_.close();
  context_.close();
}

void ZmqPublisher::publish_anomaly(const std::string& node_id,
                                    double score, double threshold,
                                    const std::vector<float>& features) {
  auto payload = pack_anomaly(node_id, score, threshold, features);

  zmq::message_t topic_msg(topic_.size());
  std::memcpy(topic_msg.data(), topic_.data(), topic_.size());
  socket_.send(topic_msg, zmq::send_flags::sndmore);

  zmq::message_t data_msg(payload.size());
  std::memcpy(data_msg.data(), payload.data(), payload.size());
  socket_.send(data_msg, zmq::send_flags::none);
}

} // namespace fspm
