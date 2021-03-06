//
//  ViewController.swift
//  BetterRest
//
//  Created by Joseph Veverka on 8/28/20.
//  Copyright © 2020 Joseph Veverka. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    // MARK: - UIViews
    var wakeUpTime: UIDatePicker!
    var sleepAmountTime: UIStepper!
    var sleepAmountLabel: UILabel!
    var coffeeAmountStepper: UIStepper!
    var coffeeAmountLabel: UILabel!
    
    // MARK: - Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        navigationController?.navigationBar.prefersLargeTitles = true
        navigationItem.rightBarButtonItem = UIBarButtonItem(title: "Calculate", style: .plain
                                                            , target: self, action: #selector(calculateBedtime))
    }
    override func loadView() {
        // Init view
        view = UIView()
        view.backgroundColor = .white
        // Stack view
        let mainStackView = UIStackView()
        mainStackView.axis = .vertical
        mainStackView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(mainStackView)
        NSLayoutConstraint.activate([
            mainStackView.topAnchor.constraint(equalTo: view.layoutMarginsGuide.topAnchor, constant: 20),
            mainStackView.leadingAnchor.constraint(equalTo: view.layoutMarginsGuide.leadingAnchor),
            mainStackView.trailingAnchor.constraint(equalTo: view.layoutMarginsGuide.trailingAnchor)
        ])
        
        // Wake up time
        let wakeUpTitle = UILabel()
        wakeUpTitle.font = UIFont.preferredFont(forTextStyle: .headline)
        wakeUpTitle.numberOfLines = 0
        wakeUpTitle.text = "When you wanna wake up boi"
        mainStackView.addArrangedSubview(wakeUpTitle)
        
        // Variable made above
        wakeUpTime = UIDatePicker()
        wakeUpTime.datePickerMode = .time
        wakeUpTime.minuteInterval = 15
        mainStackView.addArrangedSubview(wakeUpTime)
        
        // Picker components
        var components = Calendar.current.dateComponents([.hour, .minute], from: Date())
        components.hour = 8
        components.minute = 0
        wakeUpTime.date = Calendar.current.date(from: components) ?? Date()
        
        let sleepTitle = UILabel()
        sleepTitle.font = UIFont.preferredFont(forTextStyle: .headline)
        sleepTitle.numberOfLines = 0
        sleepTitle.text = "Minimum amount of sleep?"
        mainStackView.addArrangedSubview(sleepTitle)
        
        sleepAmountTime = UIStepper()
        sleepAmountTime.addTarget(self, action: #selector(sleepAmountChanged), for: .valueChanged)
        sleepAmountTime.stepValue = 0.25
        sleepAmountTime.value = 8
        sleepAmountTime.minimumValue = 4
        sleepAmountTime.maximumValue = 12
        
        sleepAmountLabel = UILabel()
        sleepAmountLabel.font = UIFont.preferredFont(forTextStyle: .body)
        
        let sleepStackView = UIStackView()
        sleepStackView.spacing = 20
        sleepStackView.addArrangedSubview(sleepAmountTime)
        sleepStackView.addArrangedSubview(sleepAmountLabel)
        mainStackView.addArrangedSubview(sleepStackView)
        
        let coffeeTitle = UILabel()
        coffeeTitle.font = UIFont.preferredFont(forTextStyle: .headline)
        coffeeTitle.text = "How much coffee do you drink a day?"
        coffeeTitle.numberOfLines = 0
        mainStackView.addArrangedSubview(coffeeTitle)
        
        coffeeAmountStepper = UIStepper()
        coffeeAmountStepper.addTarget(self, action: #selector(coffeeAmountChanged), for: .valueChanged)
        coffeeAmountStepper.minimumValue = 1
        coffeeAmountStepper.maximumValue = 20
        
        coffeeAmountLabel = UILabel()
        coffeeAmountLabel.font = UIFont.preferredFont(forTextStyle: .body)
        
        let coffeeStackView = UIStackView()
        coffeeStackView.spacing = 20
        coffeeStackView.addArrangedSubview(coffeeAmountStepper)
        coffeeStackView.addArrangedSubview(coffeeAmountLabel)
        mainStackView.addArrangedSubview(coffeeStackView)
        
        mainStackView.setCustomSpacing(10, after: sleepTitle)
        mainStackView.setCustomSpacing(20, after: sleepStackView)
        mainStackView.setCustomSpacing(10, after: coffeeTitle)
        
        sleepAmountChanged()
        coffeeAmountChanged()
    }
    
    //MARK: - Selectors
    @objc func sleepAmountChanged() {
        sleepAmountLabel.text = String(format: "%g hours", sleepAmountTime.value)
    }
    
    @objc func coffeeAmountChanged() {
        
        if coffeeAmountStepper.value == 1 {
            coffeeAmountLabel.text = "1 cup"
        } else {
            coffeeAmountLabel.text = "\(Int(coffeeAmountStepper.value)) cups"
        }
    }
    
    // Calculate Your bed time button
    @objc func calculateBedtime() {
        
        let model = sleepcalculator()
        let title: String
        let message: String
        
        do {
            let components = Calendar.current.dateComponents([.hour, .minute], from: wakeUpTime.date)
            let hour = (components.hour ?? 0) * 60 * 60
            let minutes = (components.minute ?? 0) * 60
            
            let prediction = try model.prediction(coffee: coffeeAmountStepper.value, estimatedSleep: sleepAmountTime.value, wake: Double(hour + minutes))
            
            let formatter = DateFormatter()
            formatter.timeStyle = .short
            
            let wakeDate = wakeUpTime.date - prediction.actualSleep
            message = formatter.string(from: wakeDate)
            title = "Your ideal bedtime is..."
        } catch {
            title = "Error"
            message = "Sorry"
        }
        let ac = UIAlertController(title: title, message: message, preferredStyle: .alert)
        ac.addAction(UIAlertAction(title: "Ok", style: .default))
        present(ac, animated: true)
    }
}
