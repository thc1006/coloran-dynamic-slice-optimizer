#!/bin/bash

# Simple rollback script for Kubernetes blue-green deployments
# This script assumes you have two deployments: coloran-optimizer-blue and coloran-optimizer-green
# And a service: coloran-optimizer-service

CURRENT_ACTIVE_VERSION=$(kubectl get service coloran-optimizer-service -o jsonpath='{.spec.selector.version}')

if [ "$CURRENT_ACTIVE_VERSION" == "blue" ]; then
  echo "Currently active version is blue. Rolling back to green..."
  kubectl patch service coloran-optimizer-service -p '{"spec":{"selector":{"version":"green"}}}'
  echo "Service selector updated to green. Rollback initiated."
elif [ "$CURRENT_ACTIVE_VERSION" == "green" ]; then
  echo "Currently active version is green. Rolling back to blue..."
  kubectl patch service coloran-optimizer-service -p '{"spec":{"selector":{"version":"blue"}}}'
  echo "Service selector updated to blue. Rollback initiated."
else
  echo "Could not determine current active version. Manual intervention may be required."
  exit 1
fi

echo "Rollback script finished."
