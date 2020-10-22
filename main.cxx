#include <NumCpp.hpp>
#include <vector>
#include <map>
#include <string>
#include <iostream>

static float
softmax(nc::NdArray<float>& w, nc::NdArray<float>& x) {
  auto v = nc::dot<float>(w, x).item();
  return 1.0 / (1.0 + std::exp(-v));
}

static float
predict(nc::NdArray<float>& w, nc::NdArray<float>& x) {
  return softmax(w, x);
}

static nc::NdArray<float>
logistic_regression(nc::NdArray<float>& X, nc::NdArray<float>& y, float rate, int ntrains) {
  auto w = nc::random::randN<float>({1, X.numCols()});

  for (auto n = 0; n < ntrains; n++) {
    for (size_t i = 0; i < X.numRows(); i++) {
      auto&& x = X.row(i);
      auto&& t = nc::NdArray<float>(x.copy());
      auto pred = softmax(t, w);
      auto perr = y.at(0, i) - pred;
      auto scale = rate * perr * pred * (1 - pred);
      auto&& dx = x.copy();
      dx += x;
      dx *= scale;
      w += dx;
    }
  }

  return w;
}

static std::vector<std::string>
split(std::string& fname, char delimiter) {
  std::istringstream f(fname);
  std::string field;
  std::vector<std::string> result;
  while (getline(f, field, delimiter)) {
    result.push_back(field);
  }
  return result;
}

int main() {
  std::ifstream ifs("iris.csv");

  std::string line;

  // skip header
  std::getline(ifs, line);

  std::vector<float> rows;
  std::vector<std::string> names;
  while (std::getline(ifs, line)) {
    // sepal length, sepal width, petal length, petal width, name
    auto cells = split(line, ',');
    rows.push_back(std::stof(cells.at(0)));
    rows.push_back(std::stof(cells.at(1)));
    rows.push_back(std::stof(cells.at(2)));
    rows.push_back(std::stof(cells.at(3)));
    names.push_back(cells.at(4));
  }
  // make vector 4 dimentioned
  nc::NdArray<float> X(rows);
  X.reshape(rows.size()/4, 4);

  // make onehot values of names
  std::map<std::string, int> labels;
  for(auto& name : names) {
    if (labels.count(name) == 0) labels[name] = labels.size();
  }
  std::vector<float> n;
  for (auto& name : names) {
    if (labels.count(name) > 0) n.push_back((float)labels[name]);
  }
  nc::NdArray<float> y(n);
  y /= (float) labels.size();

  names.clear();
  for(auto& k : labels) {
    names.push_back(k.first);
  }

  // make factor from input values
  auto w = logistic_regression(X, y, 0.1, 300);

  // predict samples
  for (size_t i = 0; i < X.numRows(); i++) {
    auto x = X.row(i);
    auto n = (int) (predict(w, x) * (float) labels.size() + 0.5);
    std::cout << names[n] << std::endl;
  }

  return 0;
}
